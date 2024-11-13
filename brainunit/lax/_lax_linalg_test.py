import jax.numpy as jnp
import jax.lax as lax
import pytest
from absl.testing import parameterized

import brainunit as bu
import brainunit.lax as bulax
from brainunit import meter
from brainunit._base import assert_quantity

lax_linear_algebra_unary = [
    'cholesky', 'eig', 'eigh', 'hessenberg', 'lu',
    'qdwh', 'qr', 'schur', 'svd',
    'tridiagonal',
]

lax_linear_algebra_binary = [
    'householder_product', 'triangular_solve',
]

lax_linear_algebra_nary = [
    'tridiagonal_solve',
]