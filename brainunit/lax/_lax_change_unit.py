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
from typing import Callable, Union, Sequence

import jax
from jax import lax

from .._base import Quantity, maybe_decimal, UNITLESS
from .._misc import set_module_as
from ..math._fun_change_unit import _fun_change_unit_unary, _fun_change_unit_binary

__all__ = [
    # math funcs change unit (unary)
    'rsqrt',

    # math funcs change unit (binary)
    'conv', 'conv_transpose', 'div', 'dot_general',
    'pow', 'integer_pow', 'mul', 'rem', 'batch_matmul',
]


def unit_change(
        unit_change_fun: Callable
):
    def actual_decorator(func):
        func._unit_change_fun = unit_change_fun
        return set_module_as('brainunit.lax')(func)

    return actual_decorator


# math funcs change unit (unary)
@unit_change(lambda u: u ** -0.5)
def rsqrt(
        x: Union[jax.typing.ArrayLike, Quantity],
) -> Union[Quantity, jax.Array]:
    return _fun_change_unit_unary(lax.rsqrt,
                                  lambda u: u ** -0.5,
                                  x)


# math funcs change unit (binary)
@unit_change(lambda x, y: x * y)
def conv(
        x: Union[jax.typing.ArrayLike, Quantity],
        y: Union[jax.typing.ArrayLike, Quantity],
        window_strides: Sequence[int],
        padding: str,
        precision: lax.PrecisionLike = None,
        preferred_element_type: jax.typing.DTypeLike | None = None
) -> Union[Quantity, jax.Array]:
    return _fun_change_unit_binary(lax.conv,
                                   lambda x, y: x * y,
                                   x, y,
                                   window_strides, padding, precision, preferred_element_type)


@unit_change(lambda x, y: x * y)
def conv_transpose(
        x: Union[jax.typing.ArrayLike, Quantity],
        y: Union[jax.typing.ArrayLike, Quantity],
        strides: Sequence[int],
        padding: str | Sequence[tuple[int, int]],
        rhs_dilation: Sequence[int] | None = None,
        dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None,
        transpose_kernel: bool = False,
        precision: lax.PrecisionLike = None,
        preferred_element_type: jax.typing.DTypeLike | None = None
) -> Union[Quantity, jax.Array]:
    return _fun_change_unit_binary(lax.conv_transpose,
                                   lambda x, y: x * y,
                                   x, y,
                                   strides, padding, rhs_dilation, dimension_numbers, transpose_kernel, precision,
                                   preferred_element_type)


@unit_change(lambda x, y: x / y)
def div(
        x: Union[jax.typing.ArrayLike, Quantity],
        y: Union[jax.typing.ArrayLike, Quantity],
) -> Union[Quantity, jax.Array]:
    return _fun_change_unit_binary(lax.div,
                                   lambda x, y: x / y,
                                   x, y)


@unit_change(lambda x, y: x * y)
def dot_general(
        x: Union[jax.typing.ArrayLike, Quantity],
        y: Union[jax.typing.ArrayLike, Quantity],
        dimension_numbers: jax.lax.DotDimensionNumbers,
        precision: jax.lax.PrecisionLike = None,
        preferred_element_type: jax.typing.DTypeLike | None = None,
        out_type=None
) -> Union[Quantity, jax.Array]:
    return _fun_change_unit_binary(lax.dot_general,
                                   lambda x, y: x * y,
                                   x, y,
                                   dimension_numbers, precision, preferred_element_type, out_type)


@set_module_as('brainunit.lax')
def pow(
        x: Union[Quantity, jax.typing.ArrayLike],
        y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    if isinstance(x, Quantity):
        if isinstance(y, Quantity):
            assert y.is_unitless, f'{jax.lax.pow.__name__} only supports scalar exponent'
            y = y.mantissa
        return maybe_decimal(Quantity(jax.lax.pow(x.mantissa, y), unit=x.unit ** y))
    elif isinstance(y, Quantity):
        assert y.is_unitless, f'{jax.lax.power.__name__} only supports scalar exponent'
        y = y.mantissa
        return maybe_decimal(Quantity(jax.lax.pow(x, y), unit=x ** y))
    else:
        return jax.lax.pow(x, y)


@set_module_as('brainunit.lax')
def integer_pow(
        x: Union[Quantity, jax.typing.ArrayLike],
        y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    if isinstance(x, Quantity):
        if isinstance(y, Quantity):
            assert y.is_unitless, f'{jax.lax.integer_pow.__name__} only supports scalar exponent'
            y = y.mantissa
        return maybe_decimal(Quantity(jax.lax.integer_pow(x.mantissa, y), unit=x.unit ** y))
    elif isinstance(y, Quantity):
        assert y.is_unitless, f'{jax.lax.integer_power.__name__} only supports scalar exponent'
        y = y.mantissa
        return maybe_decimal(Quantity(jax.lax.integer_pow(x, y), unit=x ** y))
    else:
        return jax.lax.integer_pow(x, y)


@unit_change(lambda x, y: x * y)
def mul(
        x: Union[Quantity, jax.typing.ArrayLike],
        y: Union[Quantity, jax.typing.ArrayLike]
) -> Union[Quantity, jax.typing.ArrayLike]:
    return _fun_change_unit_binary(lax.mul,
                                   lambda x, y: x * y,
                                   x, y)


@set_module_as('brainunit.lax')
def rem(
        x: Union[Quantity, jax.typing.ArrayLike],
        y: Union[Quantity, jax.typing.ArrayLike]
) -> Union[Quantity, jax.typing.ArrayLike]:
    if isinstance(x, Quantity) and isinstance(y, Quantity):
        return maybe_decimal(Quantity(lax.rem(x.mantissa, y.mantissa), unit=x.unit))
    elif isinstance(x, Quantity):
        return maybe_decimal(Quantity(lax.rem(x.mantissa, y), unit=x.unit))
    elif isinstance(y, Quantity):
        return maybe_decimal(Quantity(lax.rem(x, y.mantissa), unit=UNITLESS))


@unit_change(lambda x, y: x * y)
def batch_matmul(
        x: Union[Quantity, jax.typing.ArrayLike],
        y: Union[Quantity, jax.typing.ArrayLike],
        precision: jax.lax.PrecisionLike = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    return _fun_change_unit_binary(lax.batch_matmul,
                                   lambda x, y: x * y,
                                   x, y, precision)


