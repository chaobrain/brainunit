import jax.numpy as jnp
import jax.lax as lax
import pytest
from absl.testing import parameterized

import brainunit as bu
import brainunit.lax as bulax
from brainunit import meter
from brainunit._base import assert_quantity

lax_array_creation_given_array = [
    'zeros_like_array',
    ]

lax_array_creation_misc = [
    'iota', 'broadcasted_iota',
]