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
from typing import Union, Optional, Callable, Sequence

import jax
from jax import lax

from .._base import Quantity, Unit
from .._misc import set_module_as
from ..math._fun_accept_unitless import _fun_accept_unitless_unary, _fun_accept_unitless_binary, _fun_unitless_binary

__all__ = [
    # math funcs only accept unitless (unary)
    'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh',
    'collapse', 'cumlogsumexp',
    'bessel_i0e', 'bessel_i1e', 'digamma', 'lgamma', 'erf', 'erfc',
    'erf_inv', 'logistic',

    # math funcs only accept unitless (binary)
    'atan2', 'polygamma', 'igamma', 'igammac', 'igamma_grad_a', 'random_gamma_grad',
    'zeta',

    # math funcs only accept unitless (n-ary)
    'betainc', 'betainc_gradx', 'igamma_gradx',
    'igamma_grada', 'igammac_gradx', 'igammac_grada', 'polygamma_gradx',

    # Elementwise bit operations (unary)

    # Elementwise bit operations (binary)
    'shift_left', 'shift_right_arithmetic', 'shift_right_logical',

    # fft
    'fft',
]


# math funcs only accept unitless (unary)
# ---------------------------------------

@set_module_as('brainunit.lax')
def acos(
        x: Union[Quantity, jax.typing.ArrayLike],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_unary(lax.acos, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def acosh(
        x: Union[Quantity, jax.typing.ArrayLike],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_unary(lax.acosh, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def asin(
        x: Union[Quantity, jax.typing.ArrayLike],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_unary(lax.asin, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def asinh(
        x: Union[Quantity, jax.typing.ArrayLike],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_unary(lax.asinh, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def atan(
        x: Union[Quantity, jax.typing.ArrayLike],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_unary(lax.atan, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def atanh(
        x: Union[Quantity, jax.typing.ArrayLike],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_unary(lax.atanh, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def collapse(
        x: Union[Quantity, jax.typing.ArrayLike],
        start_dimension: int,
        end_dimension: Optional[int] = None,
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_unary(lax.collapse, x, start_dimension, end_dimension,
                                      unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def cumlogsumexp(
        x: Union[Quantity, jax.typing.ArrayLike],
        axis: Optional[int] = 0,
        reverse: Optional[bool] = False,
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_unary(lax.cumlogsumexp, x, axis, reverse,
                                      unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def bessel_i0e(
        x: Union[Quantity, jax.typing.ArrayLike],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_unary(lax.bessel_i0e, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def bessel_i1e(
        x: Union[Quantity, jax.typing.ArrayLike],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_unary(lax.bessel_i1e, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def digamma(
        x: Union[Quantity, jax.typing.ArrayLike],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_unary(lax.digamma, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def lgamma(
        x: Union[Quantity, jax.typing.ArrayLike],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_unary(lax.lgamma, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def erf(
        x: Union[Quantity, jax.typing.ArrayLike],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_unary(lax.erf, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def erfc(
        x: Union[Quantity, jax.typing.ArrayLike],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_unary(lax.erfc, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def erf_inv(
        x: Union[Quantity, jax.typing.ArrayLike],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_unary(lax.erf_inv, x, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def logistic(
        x: Union[Quantity, jax.typing.ArrayLike],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_unary(lax.logistic, x, unit_to_scale=unit_to_scale)


# math funcs only accept unitless (binary)
# ----------------------------------------
@set_module_as('brainunit.lax')
def atan2(
        x: Union[jax.typing.ArrayLike, Quantity],
        y: Union[jax.typing.ArrayLike, Quantity],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_binary(lax.atan2, x, y, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def polygamma(
        x: Union[jax.typing.ArrayLike, Quantity],
        y: Union[jax.typing.ArrayLike, Quantity],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_binary(lax.polygamma, x, y, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def igamma(
        x: Union[jax.typing.ArrayLike, Quantity],
        y: Union[jax.typing.ArrayLike, Quantity],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_binary(lax.igamma, x, y, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def igammac(
        x: Union[jax.typing.ArrayLike, Quantity],
        y: Union[jax.typing.ArrayLike, Quantity],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_binary(lax.igammac, x, y, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def igamma_grad_a(
        x: Union[jax.typing.ArrayLike, Quantity],
        y: Union[jax.typing.ArrayLike, Quantity],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_binary(lax.igamma_grad_a, x, y, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def random_gamma_grad(
        x: Union[jax.typing.ArrayLike, Quantity],
        y: Union[jax.typing.ArrayLike, Quantity],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_binary(lax.random_gamma_grad, x, y, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def zeta(
        x: Union[jax.typing.ArrayLike, Quantity],
        q: Union[jax.typing.ArrayLike, Quantity],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_binary(lax.zeta, x, q, unit_to_scale=unit_to_scale)


# math funcs only accept unitless (n-ary)
# ---------------------------------------

def _fun_accept_unitless_nary(
        func: Callable,
        *args,
        quantity_num: int,
        unit_to_scale: Optional[Unit] = None,
        **kwargs,
):
    if not isinstance(quantity_num, int):
        raise TypeError(f'quantity_num should be an integer. Got {quantity_num}')
    for arg in args:
        if isinstance(arg, Quantity):
            if unit_to_scale is None:
                assert arg.dim.is_dimensionless, (
                    f'{func} only support dimensionless input. But we got {arg}. \n'
                    f'If you want to scale the input, please provide the "unit_to_scale" parameter. Or '
                    f'convert the input to a dimensionless Quantity manually.'
                )
                arg = arg.to_decimal()
            else:
                assert isinstance(unit_to_scale, Unit), f'unit_to_scale should be a Unit instance. Got {unit_to_scale}'
                arg = arg.to_decimal(unit_to_scale)
    return func(*args, **kwargs)


@set_module_as('brainunit.lax')
def betainc(
        a: Union[jax.typing.ArrayLike, Quantity],
        b: Union[jax.typing.ArrayLike, Quantity],
        x: Union[jax.typing.ArrayLike, Quantity],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_nary(lax.betainc, a, b, x,
                                     quantity_num=3, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def betainc_gradx(
        g: Union[jax.typing.ArrayLike, Quantity],
        a: Union[jax.typing.ArrayLike, Quantity],
        b: Union[jax.typing.ArrayLike, Quantity],
        x: Union[jax.typing.ArrayLike, Quantity],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_nary(lax.betainc_gradx, g, a, b, x,
                                     quantity_num=4, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def igamma_gradx(
        g: Union[jax.typing.ArrayLike, Quantity],
        a: Union[jax.typing.ArrayLike, Quantity],
        x: Union[jax.typing.ArrayLike, Quantity],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_nary(lax.igamma_gradx, g, a, x,
                                     quantity_num=3, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def igamma_grada(
        g: Union[jax.typing.ArrayLike, Quantity],
        a: Union[jax.typing.ArrayLike, Quantity],
        x: Union[jax.typing.ArrayLike, Quantity],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_nary(lax.igamma_grada, g, a, x,
                                     quantity_num=3, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def igammac_gradx(
        g: Union[jax.typing.ArrayLike, Quantity],
        a: Union[jax.typing.ArrayLike, Quantity],
        x: Union[jax.typing.ArrayLike, Quantity],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_nary(lax.igammac_gradx, g, a, x,
                                     quantity_num=3, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def igammac_grada(
        g: Union[jax.typing.ArrayLike, Quantity],
        a: Union[jax.typing.ArrayLike, Quantity],
        x: Union[jax.typing.ArrayLike, Quantity],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_nary(lax.igammac_grada, g, a, x,
                                     quantity_num=3, unit_to_scale=unit_to_scale)


@set_module_as('brainunit.lax')
def polygamma_gradx(
        g: Union[jax.typing.ArrayLike, Quantity],
        m: Union[jax.typing.ArrayLike, Quantity],
        x: Union[jax.typing.ArrayLike, Quantity],
        unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    return _fun_accept_unitless_nary(lax.polygamma_gradx, g, m, x,
                                     quantity_num=3, unit_to_scale=unit_to_scale)


# Elementwise bit operations (binary)
@set_module_as('brainunit.lax')
def shift_left(
        x: Union[Quantity, jax.typing.ArrayLike],
        y: Union[Quantity, jax.typing.ArrayLike]
) -> jax.Array:
    return _fun_unitless_binary(lax.shift_left, x, y)


@set_module_as('brainunit.lax')
def shift_right_arithmetic(
        x: Union[Quantity, jax.typing.ArrayLike],
        y: Union[Quantity, jax.typing.ArrayLike]
) -> jax.Array:
    return _fun_unitless_binary(lax.shift_right_arithmetic, x, y)


@set_module_as('brainunit.lax')
def shift_right_logical(
        x: Union[Quantity, jax.typing.ArrayLike],
        y: Union[Quantity, jax.typing.ArrayLike]
) -> jax.Array:
    return _fun_unitless_binary(lax.shift_right_logical, x, y)


# fft
@set_module_as('brainunit.lax')
def fft(
        x: Union[Quantity, jax.typing.ArrayLike],
        fft_type: jax.lax.FftType | str,
        fft_lengths: Sequence[int],
        unit_to_scale: Optional[Unit] = None,
):
    return _fun_accept_unitless_unary(lax.fft, x, fft_type, fft_lengths,
                                      unit_to_scale=unit_to_scale)
