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

from typing import Optional, Union, Sequence

import jax
from jax import lax

from brainunit._misc import set_module_as
from .._base import Unit, Quantity

Shape = Union[int, Sequence[int]]

__all__ = [
    # array creation(given shape)

    # array creation(given array)
    'zeros_like_array',

    # array creation(misc)
    'iota', 'broadcasted_iota', 'zeros_like_shaped_array',

    # indexing funcs

    # others

]


# array creation (given array)
@set_module_as('brainunit.lax')
def zeros_like_array(
        x: Union[Quantity, jax.typing.ArrayLike],
        unit: Optional[Unit] = None,
) -> Union[Quantity, jax.Array]:
    if isinstance(x, Quantity):
        if unit is not None:
            assert isinstance(unit, Unit), 'unit must be an instance of Unit.'
            x = x.in_unit(unit)
        return Quantity(lax.zeros_like_array(x.mantissa), unit=x.unit)
    else:
        if unit is not None:
            assert isinstance(unit, Unit), 'unit must be an instance of Unit.'
            return lax.zeros_like_array(x) * unit
        else:
            return lax.zeros_like_array(x)


# array creation (misc)
@set_module_as('brainunit.lax')
def iota(
        dtype: jax.typing.DTypeLike,
        size: int,
        unit: Optional[Unit] = None,
) -> Union[Quantity, jax.Array]:
    if unit is not None:
        assert isinstance(unit, Unit), 'unit must be an instance of Unit.'
        return lax.iota(dtype, size) * unit
    else:
        return lax.iota(dtype, size)


@set_module_as('brainunit.lax')
def broadcasted_iota(
        dtype: jax.typing.DTypeLike,
        shape: Shape,
        dimension: int,
        _sharding=None,
        unit: Optional[Unit] = None,
) -> Union[Quantity, jax.Array]:
    if unit is not None:
        assert isinstance(unit, Unit), 'unit must be an instance of Unit.'
        return lax.broadcasted_iota(dtype, shape, dimension, _sharding) * unit
    else:
        return lax.broadcasted_iota(dtype, shape, dimension, _sharding)


def zeros_like_shaped_array(
        aval: jax.core.ShapedArray,
        unit: Optional[Unit] = None,
):
    if unit is not None:
        assert isinstance(unit, Unit), 'unit must be an instance of Unit.'
        return lax.zeros_like_shaped_array(aval) * unit
    else:
        return lax.zeros_like_shaped_array(aval)