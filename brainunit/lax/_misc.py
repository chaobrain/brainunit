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

from typing import Any, Callable, Sequence, Union

import jax
from jax import lax

from .. import maybe_decimal
from .._base import Quantity
from .._misc import set_module_as

__all__ = [
    'after_all', 'reduce', 'reduce_precision',

    # getting attribute funcs
    'broadcast_shapes',

    # convolution
    'conv_dimension_numbers', 'conv_general_dilated', 'conv_general_dilated_local', 'conv_general_dilated_patches',
    'conv_with_general_padding',

    # custom gradient operators
    'stop_gradient', 'custom_linear_solve', 'custom_root',

    # sharding-related operators
    'with_sharding_constraint',
]


@set_module_as('brainunit.lax')
def after_all(*operands):
    new_operands = []
    for operand in operands:
        if isinstance(operand, Quantity):
            new_operands.append(operand.mantissa)
        else:
            new_operands.append(operand)
    return lax.after_all(*new_operands)


@set_module_as('brainunit.lax')
def reduce(
        operands: Any,
        init_values: Any,
        computation: Callable[[Any, Any], Any],
        dimensions: Sequence[int]
) -> Any:
    return lax.reduce(operands, init_values, computation, dimensions)


def reduce_precision(
        operand: Union[jax.typing.ArrayLike, Quantity, float],
        exponent_bits: int,
        mantissa_bits: int
) -> jax.typing.ArrayLike:
    if isinstance(operand, Quantity):
        return maybe_decimal(lax.reduce_precision(operand.mantissa, exponent_bits, mantissa_bits))
    return lax.reduce_precision(operand, exponent_bits, mantissa_bits)


@set_module_as('brainunit.lax')
def broadcast_shapes(
        *shapes
):
    return lax.broadcast_shapes(*shapes)