from typing import Union, Callable, Any

import jax
from jax import lax

from brainunit.lax._lax_change_unit import unit_change
from .._base import Quantity, maybe_decimal, fail_for_unit_mismatch
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
        a: Union[Quantity, jax.typing.ArrayLike],
        taus: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    # TODO: more proper handling of Quantity?
    if isinstance(a, Quantity) and isinstance(taus, Quantity):
        return lax.linalg.householder_product(a.mantissa, taus.mantissa)
    elif isinstance(a, Quantity):
        return lax.linalg.householder_product(a.mantissa, taus)
    elif isinstance(taus, Quantity):
        return lax.linalg.householder_product(a, taus.mantissa)
    else:
        return lax.linalg.householder_product(a, taus)


@set_module_as('brainunit.lax')
def qdwh(
        x: Union[Quantity, jax.typing.ArrayLike],
) -> tuple[jax.Array, Quantity | jax.Array, int, bool]:
    if isinstance(x, Quantity):
        u, h, num_iters, is_converged = lax.linalg.qdwh(x.mantissa)
        return u, maybe_decimal(Quantity(h, unit=x.unit)), num_iters, is_converged
    else:
        return lax.linalg.qdwh(x)


@set_module_as('brainunit.lax')
def qr(
        x: Union[Quantity, jax.typing.ArrayLike],
) -> tuple[jax.Array, Quantity | jax.Array]:
    if isinstance(x, Quantity):
        q, r = lax.linalg.qr(x.mantissa)
        return q, maybe_decimal(Quantity(r, unit=x.unit))
    else:
        return lax.linalg.qr(x)


@set_module_as('brainunit.lax')
def schur(
        x: Union[Quantity, jax.typing.ArrayLike],
        compute_schur_vectors: bool = True,
        sort_eig_vals: bool = False,
        select_callable: Callable[..., Any] | None = None
) -> tuple[jax.Array, Quantity | jax.Array]:
    if isinstance(x, Quantity):
        t, q = lax.linalg.schur(x.mantissa, compute_schur_vectors=compute_schur_vectors,
                                sort_eig_vals=sort_eig_vals, select_callable=select_callable)
        return t, maybe_decimal(Quantity(q, unit=x.unit))
    else:
        return lax.linalg.schur(x, compute_schur_vectors=compute_schur_vectors,
                                sort_eig_vals=sort_eig_vals, select_callable=select_callable)


@set_module_as('brainunit.lax')
def svd(
        x: Union[Quantity, jax.typing.ArrayLike],
) -> tuple[jax.Array, Quantity | jax.Array, jax.Array]:
    if isinstance(x, Quantity):
        u, s, vh = lax.linalg.svd(x.mantissa)
        return u, maybe_decimal(Quantity(s, unit=x.unit)), vh
    else:
        return lax.linalg.svd(x)


@set_module_as('brainunit.lax')
def triangular_solve(
        a: Union[Quantity, jax.typing.ArrayLike],
        b: Union[Quantity, jax.typing.ArrayLike],
        left_side: bool = False, lower: bool = False,
        transpose_a: bool = False, conjugate_a: bool = False,
        unit_diagonal: bool = False,
) -> Quantity | jax.Array:
    if isinstance(a, Quantity) and isinstance(b, Quantity):
        return maybe_decimal(Quantity(lax.linalg.triangular_solve(a.mantissa, b.mantissa, left_side=left_side,
                                                                 lower=lower, transpose_a=transpose_a, conjugate_a=conjugate_a,
                                                                 unit_diagonal=unit_diagonal), unit=b.unit))
    elif isinstance(a, Quantity):
        return lax.linalg.triangular_solve(a.mantissa, b, left_side=left_side,
                                                                 lower=lower, transpose_a=transpose_a, conjugate_a=conjugate_a,
                                                                 unit_diagonal=unit_diagonal)
    elif isinstance(b, Quantity):
        return maybe_decimal(Quantity(lax.linalg.triangular_solve(a, b.mantissa, left_side=left_side,
                                                                    lower=lower, transpose_a=transpose_a, conjugate_a=conjugate_a,
                                                                    unit_diagonal=unit_diagonal), unit=b.unit))
    else:
        return lax.linalg.triangular_solve(a, b, left_side=left_side,
                                           lower=lower, transpose_a=transpose_a, conjugate_a=conjugate_a,
                                           unit_diagonal=unit_diagonal)


@set_module_as('brainunit.lax')
def tridiagonal(
        a: Union[Quantity, jax.typing.ArrayLike],
        lower: bool = True,
) -> tuple[Quantity | jax.Array, Quantity | jax.Array, Quantity | jax.Array, jax.Array]:
    if isinstance(a, Quantity):
        arr, d, e, taus = lax.linalg.tridiagonal(a.mantissa, lower=lower)
        return maybe_decimal(Quantity(a, unit=a.unit)), maybe_decimal(Quantity(d, unit=a.unit)), \
               maybe_decimal(Quantity(e, unit=a.unit)), taus
    else:
        return lax.linalg.tridiagonal(a, lower=lower)


@set_module_as('brainunit.lax')
def tridiagonal_solve(
        dl: Union[Quantity, jax.typing.ArrayLike],
        d: Union[Quantity, jax.typing.ArrayLike],
        du: Union[Quantity, jax.typing.ArrayLike],
        b: Union[Quantity, jax.typing.ArrayLike],
) -> Quantity | jax.Array:
    fail_for_unit_mismatch(dl, d)
    fail_for_unit_mismatch(dl, du)
    if isinstance(b, Quantity):
        try:
            return maybe_decimal(Quantity(lax.linalg.tridiagonal_solve(dl.mantissa, d.mantissa, du.mantissa, b.mantissa), unit=b.unit))
        except:
            return Quantity(lax.linalg.tridiagonal_solve(dl, d, du, b.mantissa), unit=b.unit)
    else:
        try:
            return lax.linalg.tridiagonal_solve(dl.mantissa, d.mantissa, du.mantissa, b)
        except:
            return lax.linalg.tridiagonal_solve(dl, d, du, b)

