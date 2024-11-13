import jax.numpy as jnp
import jax.lax as lax
import pytest
from absl.testing import parameterized

import brainunit as bu
import brainunit.lax as bulax
from brainunit import meter
from brainunit._base import assert_quantity

lax_change_unit_unary = [
    'rsqrt',
    ]

lax_change_unit_binary = [
    'conv', 'conv_transpose', 'div', 'dot_general',
    'pow', 'integer_pow', 'mul', 'rem', 'batch_matmul',
]