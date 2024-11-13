import jax.numpy as jnp
import jax.lax as lax
import pytest
from absl.testing import parameterized

import brainunit as bu
import brainunit.lax as bulax
from brainunit import meter
from brainunit._base import assert_quantity

lax_remove_unit_unary = [
    'population_count', 'clz',
]

lax_logic_funcs_binary = [
    'eq', 'ne', 'ge', 'gt', 'le', 'lt',
]

lax_indexing = [
    'argmax', 'argmin',
]