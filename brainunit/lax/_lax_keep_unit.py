# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations

import builtins
from typing import Union, Sequence, Callable

import jax
import numpy as np
from jax import lax
from jax._src.typing import Shape

from .._base import Quantity, maybe_decimal
from .._misc import set_module_as
from ..math._fun_keep_unit import _fun_keep_unit_unary, _fun_keep_unit_binary

__all__ = [
    # sequence inputs

    # array manipulation
    'slice', 'dynamic_slice', 'dynamic_update_slice', 'gather',
    'index_take', 'slice_in_dim', 'index_in_dim', 'dynamic_slice_ind_dim', 'dynamic_index_in_dim',
    'dynamic_update_slice_in_dim', 'dynamic_update_index_in_dim',
    'sort', 'sort_key_val',

    # math funcs keep unit (unary)
    'neg',
    'cummax', 'cummin', 'cumsum',
    'scatter', 'scatter_add', 'scatter_sub', 'scatter_mul', 'scatter_min', 'scatter_max', 'scatter_apply',

    # math funcs keep unit (binary)
    'sub', 'complex', 'pad',

    # math funcs keep unit (n-ary)
    'clamp',

    # type conversion
    'convert_element_type', 'bitcast_convert_type',

    # math funcs keep unit (return Quantity and index)
    'approx_max_k', 'approx_min_k', 'top_k',

    # math funcs only accept unitless (unary) can return Quantity

    # broadcasting arrays
    'broadcast', 'broadcast_in_dim', 'broadcast_to_rank',
]


# array manipulation
@set_module_as('brainunit.math')
def slice(
        operand: Union[Quantity, jax.typing.ArrayLike],
        start_indices: Sequence[int],
        limit_indices: Sequence[int],
        strides: Sequence[int] | None = None
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_unary(lax.slice, operand, start_indices, limit_indices, strides)


@set_module_as('brainunit.math')
def dynamic_slice(
        operand: Union[Quantity, jax.typing.ArrayLike],
        start_indices: jax.typing.ArrayLike | Sequence[jax.typing.ArrayLike],
        slice_sizes: Shape,
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_unary(lax.dynamic_slice, operand, start_indices, slice_sizes)


@set_module_as('brainunit.math')
def dynamic_update_slice(
        operand: Union[Quantity, jax.typing.ArrayLike],
        update: Union[Quantity, jax.typing.ArrayLike],
        start_indices: jax.typing.ArrayLike | Sequence[jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_binary(lax.dynamic_update_slice, operand, update, start_indices)


@set_module_as('brainunit.math')
def gather(
        operand: Union[Quantity, jax.typing.ArrayLike],
        start_indices: jax.typing.ArrayLike,
        dimension_numbers: jax.lax.GatherDimensionNumbers,
        slice_sizes: Shape,
        *,
        unique_indices: bool = False,
        indices_are_sorted: bool = False,
        mode: str | jax.lax.GatherScatterMode | None = None,
        fill_value: Union[Quantity, jax.typing.ArrayLike] = None
) -> Union[Quantity, jax.Array]:
    if isinstance(operand, Quantity) and isinstance(fill_value, Quantity):
        return maybe_decimal(Quantity(lax.gather(operand.value, start_indices, dimension_numbers, slice_sizes,
                                                 unique_indices=unique_indices, indices_are_sorted=indices_are_sorted,
                                                 mode=mode, fill_value=fill_value.in_unit(operand.unit)),
                                      unit=operand.unit))
    elif isinstance(operand, Quantity):
        if fill_value is not None:
            raise ValueError('fill_value must be a Quantity if operand is a Quantity')
        return maybe_decimal(Quantity(lax.gather(operand.value, start_indices, dimension_numbers, slice_sizes,
                                                 unique_indices=unique_indices, indices_are_sorted=indices_are_sorted,
                                                 mode=mode), unit=operand.unit))
    elif isinstance(fill_value, Quantity):
        raise ValueError('fill_value must be None if operand is not a Quantity')
    return lax.gather(operand, start_indices, dimension_numbers, slice_sizes,
                      unique_indices=unique_indices, indices_are_sorted=indices_are_sorted,
                      mode=mode, fill_value=fill_value)


@set_module_as('brainunit.math')
def index_take(
        src: Union[Quantity, jax.typing.ArrayLike],
        idxs: jax.typing.ArrayLike,
        axes: Sequence[int]
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_unary(lax.index_take, src, idxs, axes)


@set_module_as('brainunit.math')
def slice_in_dim(
        operand: Union[Quantity, jax.typing.ArrayLike],
        start_index: int | None,
        limit_index: int | None,
        stride: int = 1,
        axis: int = 0
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_unary(lax.slice_in_dim, operand, start_index, limit_index, stride, axis)


@set_module_as('brainunit.math')
def index_in_dim(
        operand: Union[Quantity, jax.typing.ArrayLike],
        index: int,
        axis: int = 0,
        keepdims: bool = True
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_unary(lax.index_in_dim, operand, index, axis, keepdims)


@set_module_as('brainunit.math')
def dynamic_slice_ind_dim(
        operand: Union[Quantity, jax.typing.ArrayLike],
        start_index: jax.typing.ArrayLike,
        slice_size: int,
        axis: int = 0
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_unary(lax.dynamic_slice_in_dim, operand, start_index, slice_size, axis)


@set_module_as('brainunit.math')
def dynamic_index_in_dim(
        operand: Union[Quantity, jax.typing.ArrayLike],
        index: int | jax.typing.ArrayLike,
        axis: int = 0, keepdims: bool = True
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_unary(lax.dynamic_index_in_dim, operand, index, axis, keepdims)


@set_module_as('brainunit.math')
def dynamic_update_slice_in_dim(
        operand: Union[Quantity, jax.typing.ArrayLike],
        update: Union[Quantity, jax.typing.ArrayLike],
        start_index: jax.typing.ArrayLike, axis: int
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_binary(lax.dynamic_update_slice_in_dim, operand, update, start_index, axis)


@set_module_as('brainunit.math')
def dynamic_update_index_in_dim(
        operand: Union[Quantity, jax.typing.ArrayLike],
        update: Union[Quantity, jax.typing.ArrayLike],
        index: jax.typing.ArrayLike,
        axis: int
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_binary(lax.dynamic_update_index_in_dim, operand, update, index, axis)


@set_module_as('brainunit.math')
def sort(
        operand: Union[Quantity, jax.typing.ArrayLike] | Sequence[Union[Quantity, jax.typing.ArrayLike]],
        dimension: int = -1,
        is_stable: bool = True, num_keys: int = 1
) -> Union[Quantity, jax.Array] | Sequence[Union[Quantity, jax.Array]]:
    # check if operand is a sequence
    if isinstance(operand, Sequence):
        output = []
        for op in operand:
            if isinstance(op, Quantity):
                output.append(Quantity(lax.sort(op.mantissa, dimension, is_stable, num_keys), unit=op.unit))
            else:
                output.append(lax.sort(op, dimension, is_stable, num_keys))
        return output
    else:
        if isinstance(operand, Quantity):
            return Quantity(lax.sort(operand.mantissa, dimension, is_stable, num_keys), unit=operand.unit)
        return lax.sort(operand, dimension, is_stable, num_keys)


@set_module_as('brainunit.math')
def sort_key_val(
        keys: Union[Quantity, jax.typing.ArrayLike],
        values: Union[Quantity, jax.typing.ArrayLike],
        dimension: int = -1,
        is_stable: bool = True
) -> tuple[Union[Quantity, jax.Array], Union[Quantity, jax.Array]]:
    if isinstance(keys, Quantity) and isinstance(values, Quantity):
        k, v = lax.sort_key_val(keys.mantissa, values.mantissa, dimension, is_stable)
        return maybe_decimal(Quantity(k, unit=keys.unit)), maybe_decimal(Quantity(v, unit=values.unit))
    elif isinstance(keys, Quantity):
        k, v = lax.sort_key_val(keys.mantissa, values, dimension, is_stable)
        return maybe_decimal(Quantity(k, unit=keys.unit)), v
    elif isinstance(values, Quantity):
        k, v = lax.sort_key_val(keys, values.mantissa, dimension, is_stable)
        return k, maybe_decimal(Quantity(v, unit=values.unit))
    return lax.sort_key_val(keys, values, dimension, is_stable)


# math funcs keep unit (unary)
@set_module_as('brainunit.math')
def neg(
        x: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_unary(lax.neg, x)


@set_module_as('brainunit.math')
def cummax(
        operand: Union[Quantity, jax.typing.ArrayLike],
        axis: int = 0,
        reverse: bool = False
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_unary(lax.cummax, operand, axis, reverse)


@set_module_as('brainunit.math')
def cummin(
        operand: Union[Quantity, jax.typing.ArrayLike],
        axis: int = 0,
        reverse: bool = False
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_unary(lax.cummin, operand, axis, reverse)


@set_module_as('brainunit.math')
def cumsum(
        operand: Union[Quantity, jax.typing.ArrayLike],
        axis: int = 0,
        reverse: bool = False
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_unary(lax.cumsum, operand, axis, reverse)


def _fun_lax_scatter(
        fun: Callable,
        operand,
        scatter_indices,
        updates,
        dimension_numbers,
        indices_are_sorted,
        unique_indices,
        mode
) -> Union[Quantity, jax.Array]:
    if isinstance(operand, Quantity):
        return maybe_decimal(Quantity(fun(operand.mantissa, scatter_indices, updates.mantissa, dimension_numbers,
                                          indices_are_sorted=indices_are_sorted,
                                          unique_indices=unique_indices,
                                          mode=mode), unit=operand.unit))
    else:
        return fun(operand, scatter_indices, updates, dimension_numbers,
                   indices_are_sorted=indices_are_sorted,
                   unique_indices=unique_indices,
                   mode=mode)


@set_module_as('brainunit.math')
def scatter(
        operand: Union[Quantity, jax.typing.ArrayLike],
        scatter_indices: jax.typing.ArrayLike,
        updates: jax.typing.ArrayLike,
        dimension_numbers: jax.lax.ScatterDimensionNumbers,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | jax.lax.GatherScatterMode | None = None
) -> Union[Quantity, jax.Array]:
    return _fun_lax_scatter(lax.scatter, operand, scatter_indices, updates, dimension_numbers, indices_are_sorted,
                            unique_indices, mode)


@set_module_as('brainunit.math')
def scatter_add(
        operand: Union[Quantity, jax.typing.ArrayLike],
        scatter_indices: jax.typing.ArrayLike,
        updates: jax.typing.ArrayLike,
        dimension_numbers: jax.lax.ScatterDimensionNumbers,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | jax.lax.GatherScatterMode | None = None
) -> Union[Quantity, jax.Array]:
    return _fun_lax_scatter(lax.scatter_add, operand, scatter_indices, updates, dimension_numbers, indices_are_sorted,
                            unique_indices, mode)


@set_module_as('brainunit.math')
def scatter_sub(
        operand: Union[Quantity, jax.typing.ArrayLike],
        scatter_indices: jax.typing.ArrayLike,
        updates: jax.typing.ArrayLike,
        dimension_numbers: jax.lax.ScatterDimensionNumbers,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | jax.lax.GatherScatterMode | None = None
) -> Union[Quantity, jax.Array]:
    return _fun_lax_scatter(lax.scatter_sub, operand, scatter_indices, updates, dimension_numbers, indices_are_sorted,
                            unique_indices, mode)


@set_module_as('brainunit.math')
def scatter_mul(
        operand: Union[Quantity, jax.typing.ArrayLike],
        scatter_indices: jax.typing.ArrayLike,
        updates: jax.typing.ArrayLike,
        dimension_numbers: jax.lax.ScatterDimensionNumbers,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | jax.lax.GatherScatterMode | None = None
) -> Union[Quantity, jax.Array]:
    return _fun_lax_scatter(lax.scatter_mul, operand, scatter_indices, updates, dimension_numbers, indices_are_sorted,
                            unique_indices, mode)


@set_module_as('brainunit.math')
def scatter_min(
        operand: Union[Quantity, jax.typing.ArrayLike],
        scatter_indices: jax.typing.ArrayLike,
        updates: jax.typing.ArrayLike,
        dimension_numbers: jax.lax.ScatterDimensionNumbers,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | jax.lax.GatherScatterMode | None = None
) -> Union[Quantity, jax.Array]:
    return _fun_lax_scatter(lax.scatter_min, operand, scatter_indices, updates, dimension_numbers, indices_are_sorted,
                            unique_indices, mode)


@set_module_as('brainunit.math')
def scatter_max(
        operand: Union[Quantity, jax.typing.ArrayLike],
        scatter_indices: jax.typing.ArrayLike,
        updates: jax.typing.ArrayLike,
        dimension_numbers: jax.lax.ScatterDimensionNumbers,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | jax.lax.GatherScatterMode | None = None
) -> Union[Quantity, jax.Array]:
    return _fun_lax_scatter(lax.scatter_max, operand, scatter_indices, updates, dimension_numbers, indices_are_sorted,
                            unique_indices, mode)


@set_module_as('brainunit.math')
def scatter_apply(
        operand: Union[Quantity, jax.typing.ArrayLike],
        scatter_indices: jax.typing.ArrayLike,
        func: Callable,
        dimension_numbers: jax.lax.ScatterDimensionNumbers,
        *,
        update_shape: Shape = (),
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | jax.lax.GatherScatterMode | None = None
) -> Union[Quantity, jax.Array]:
    if isinstance(operand, Quantity):
        return maybe_decimal(Quantity(lax.scatter_apply(operand.mantissa, scatter_indices, func, dimension_numbers,
                                                        update_shape=update_shape,
                                                        indices_are_sorted=indices_are_sorted,
                                                        unique_indices=unique_indices,
                                                        mode=mode), unit=operand.unit))
    else:
        return lax.scatter_apply(operand, scatter_indices, func, dimension_numbers,
                                 update_shape=update_shape,
                                 indices_are_sorted=indices_are_sorted,
                                 unique_indices=unique_indices,
                                 mode=mode)


# math funcs keep unit (binary)
@set_module_as('brainunit.math')
def complex(
        x: Union[Quantity, jax.typing.ArrayLike],
        y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_binary(lax.complex, x, y)


@set_module_as('brainunit.math')
def pad(
        operand: Union[Quantity, jax.typing.ArrayLike],
        padding_value: Union[Quantity, jax.typing.ArrayLike],
        padding_config: Sequence[tuple[int, int, int]]
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_binary(lax.pad, operand, padding_value, padding_config)


@set_module_as('brainunit.math')
def sub(
        x: Union[Quantity, jax.typing.ArrayLike],
        y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_binary(lax.sub, x, y)


# type conversion
@set_module_as('brainunit.math')
def convert_element_type(
        operand: Union[Quantity, jax.typing.ArrayLike],
        new_dtype: jax.typing.DTypeLike
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_unary(lax.convert_element_type, operand, new_dtype)


@set_module_as('brainunit.math')
def bitcast_convert_type(
        operand: Union[Quantity, jax.typing.ArrayLike],
        new_dtype: jax.typing.DTypeLike
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_unary(lax.bitcast_convert_type, operand, new_dtype)


# math funcs keep unit (n-ary)
@set_module_as('brainunit.math')
def clamp(
        min: Union[Quantity, jax.typing.ArrayLike],
        x: Union[Quantity, jax.typing.ArrayLike],
        max: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    if all(isinstance(i, Quantity) for i in (min, x, max)):
        unit = min.unit
        return maybe_decimal(Quantity(lax.clamp(min.mantissa, x.to_decimal(unit), max.to_decimal(unit)), unit=unit))
    elif all(isinstance(i, (jax.Array, np.ndarray, np.bool_, np.number, bool, int, float, builtins.complex)) for i in
             (min, x, max)):
        return lax.clamp(min, x, max)
    else:
        raise ValueError('All inputs must be Quantity or jax.typing.ArrayLike')


# math funcs keep unit (return Quantity and index)
@set_module_as('brainunit.math')
def approx_max_k(
        operand: Union[Quantity, jax.typing.ArrayLike],
        k: int,
        reduction_dimension: int = -1,
        recall_target: float = 0.95,
        reduction_input_size_override: int = -1,
        aggregate_to_topk: bool = True
) -> tuple[Union[Quantity, jax.Array], jax.typing.ArrayLike]:
    if isinstance(operand, Quantity):
        r = lax.approx_max_k(operand.mantissa, k, reduction_dimension, recall_target, reduction_input_size_override,
                             aggregate_to_topk)
        return maybe_decimal(Quantity(r[0], unit=operand.unit)), r[1]
    return lax.approx_max_k(operand, k, reduction_dimension, recall_target, reduction_input_size_override,
                            aggregate_to_topk)


@set_module_as('brainunit.math')
def approx_min_k(
        operand: Union[Quantity, jax.typing.ArrayLike],
        k: int,
        reduction_dimension: int = -1,
        recall_target: float = 0.95,
        reduction_input_size_override: int = -1,
        aggregate_to_topk: bool = True
) -> tuple[Union[Quantity, jax.Array], jax.typing.ArrayLike]:
    if isinstance(operand, Quantity):
        r = lax.approx_min_k(operand.mantissa, k, reduction_dimension, recall_target, reduction_input_size_override,
                             aggregate_to_topk)
        return maybe_decimal(Quantity(r[0], unit=operand.unit)), r[1]
    return lax.approx_min_k(operand, k, reduction_dimension, recall_target, reduction_input_size_override,
                            aggregate_to_topk)


@set_module_as('brainunit.math')
def top_k(
        operand: Union[Quantity, jax.typing.ArrayLike],
        k: int
) -> tuple[Union[Quantity, jax.Array], jax.typing.ArrayLike]:
    if isinstance(operand, Quantity):
        r = lax.top_k(operand.mantissa, k)
        return maybe_decimal(Quantity(r[0], unit=operand.unit)), r[1]
    return lax.top_k(operand, k)


# broadcasting arrays
def broadcast(
        operand: Union[Quantity, jax.typing.ArrayLike],
        sizes: Sequence[int]
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_unary(lax.broadcast, operand, sizes)


def broadcast_in_dim(
        operand: Union[Quantity, jax.typing.ArrayLike],
        shape: Shape,
        broadcast_dimensions: Sequence[int]
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_unary(lax.broadcast_in_dim, operand, shape, broadcast_dimensions)


def broadcast_to_rank(
        x: Union[Quantity, jax.typing.ArrayLike],
        rank: int
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_unary(lax.broadcast_to_rank, x, rank)
