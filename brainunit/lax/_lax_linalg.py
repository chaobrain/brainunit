from typing import Union

import jax
from jax import lax

from brainunit.lax._lax_change_unit import unit_change
from .._base import Quantity, maybe_decimal
from .._misc import set_module_as
from ..math._fun_change_unit import _fun_change_unit_unary

__all__ = [
    # linear algebra
    'cholesky', 'eig', 'eigh', 'hessenberg', 'lu',
    'householder_product', 'qdwh', 'qr', 'schur', 'svd', 'triangular_solve',
    'tridiagonal', 'tridiagonal_solve',
]


# linear algebra
@unit_change(lambda x: x ** 0.5)
def cholesky(
        x: Union[Quantity, jax.typing.ArrayLike],
        symmetrize_input: bool = True,
) -> Union[Quantity, jax.typing.ArrayLike]:
    return _fun_change_unit_unary(lax.linalg.cholesky,
                                  lambda u: u ** 0.5,
                                  x)


@set_module_as('brainunit.lax')
def eig(
        x: Union[Quantity, jax.typing.ArrayLike],
        compute_left_eigenvectors: bool = True,
        compute_right_eigenvectors: bool = True
) -> tuple[Quantity, jax.Array, jax.Array] | list[jax.Array] | tuple[Quantity, jax.Array] | Quantity:
    if compute_left_eigenvectors and compute_right_eigenvectors:
        if isinstance(x, Quantity):
            w, vl, vr = lax.linalg.eig(x.mantissa, compute_left_eigenvectors=True, compute_right_eigenvectors=True)
            return maybe_decimal(Quantity(w, unit=x.unit)), vl, vr
        else:
            return lax.linalg.eig(x, compute_left_eigenvectors=True, compute_right_eigenvectors=True)
    elif compute_left_eigenvectors:
        if isinstance(x, Quantity):
            w, vl = lax.linalg.eig(x.mantissa, compute_left_eigenvectors=True, compute_right_eigenvectors=False)
            return maybe_decimal(Quantity(w, unit=x.unit)), vl
        else:
            return lax.linalg.eig(x, compute_left_eigenvectors, compute_left_eigenvectors=True,
                                  compute_right_eigenvectors=False)

    elif compute_right_eigenvectors:
        if isinstance(x, Quantity):
            w, vr = lax.linalg.eig(x.mantissa, compute_left_eigenvectors=False, compute_right_eigenvectors=True)
            return maybe_decimal(Quantity(w, unit=x.unit)), vr
        else:
            return lax.linalg.eig(x, compute_right_eigenvectors, compute_left_eigenvectors=False,
                                  compute_right_eigenvectors=True)
    else:
        if isinstance(x, Quantity):
            w = lax.linalg.eig(x.mantissa, compute_left_eigenvectors=False, compute_right_eigenvectors=False)
            return maybe_decimal(Quantity(w, unit=x.unit))
        else:
            return lax.linalg.eig(x, compute_left_eigenvectors=False, compute_right_eigenvectors=False)


@set_module_as('brainunit.lax')
def eigh(
        x: Union[Quantity, jax.typing.ArrayLike],
        lower: bool = True,
        symmetrize_input: bool = True,
        sort_eigenvalues: bool = True,
        subset_by_index: tuple[int, int] | None = None,
) -> tuple[Quantity | jax.Array, jax.Array]:
    if isinstance(x, Quantity):
        w, v = lax.linalg.eigh(x.mantissa, lower=lower, symmetrize_input=symmetrize_input,
                               sort_eigenvalues=sort_eigenvalues, subset_by_index=subset_by_index)
        return maybe_decimal(Quantity(w, unit=x.unit)), v
    else:
        return lax.linalg.eigh(x, lower=lower, symmetrize_input=symmetrize_input,
                               sort_eigenvalues=sort_eigenvalues, subset_by_index=subset_by_index)


@set_module_as('brainunit.lax')
def hessenberg(
        x: Union[Quantity, jax.typing.ArrayLike],
) -> tuple[Quantity | jax.Array, jax.Array]:
    if isinstance(x, Quantity):
        h, q = lax.linalg.hessenberg(x.mantissa)
        return maybe_decimal(Quantity(h, unit=x.unit)), q
    else:
        return lax.linalg.hessenberg(x)


@set_module_as('brainunit.lax')
def lu(
        x: Union[Quantity, jax.typing.ArrayLike],
) -> tuple[Quantity | jax.Array, jax.Array, jax.Array]:
    if isinstance(x, Quantity):
        p, l, u = lax.linalg.lu(x.mantissa)
        return maybe_decimal(Quantity(p, unit=x.unit)), l, u
    else:
        return lax.linalg.lu(x)


@set_module_as('brainunit.lax')
def householder_product(
        x
): pass


@set_module_as('brainunit.lax')
def qdwh(x): pass


@set_module_as('brainunit.lax')
def qr(x): pass


@set_module_as('brainunit.lax')
def schur(x): pass


@set_module_as('brainunit.lax')
def svd(x): pass


@set_module_as('brainunit.lax')
def triangular_solve(x): pass


@set_module_as('brainunit.lax')
def tridiagonal(x): pass


@set_module_as('brainunit.lax')
def tridiagonal_solve(x): pass


# fft
def fft(x): pass
