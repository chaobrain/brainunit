import jax.numpy as jnp
import jax.lax as lax
import pytest
from absl.testing import parameterized

import brainunit as bu
import brainunit.lax as bulax
from brainunit import meter
from brainunit._base import assert_quantity

# math funcs only accept unitless (unary)
lax_accept_unitless_unary = ['acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh',
    'collapse', 'cumlogsumexp',
    'bessel_i0e', 'bessel_i1e', 'digamma', 'lgamma', 'erf', 'erfc',
    'erf_inv', 'logistic',]

# math funcs only accept unitless (binary)
lax_accept_unitless_binary = ['atan2', 'polygamma', 'igamma', 'igammac', 'igamma_grad_a', 'random_gamma_grad',
    'zeta',]

# math funcs only accept unitless (n-ary)
lax_accept_unitless_nary = ['betainc',]

# Elementwise bit operations (binary)
lax_bit_operation_binary = ['shift_left', 'shift_right_arithmetic', 'shift_right_logical',]

# fft
lax_fft = ['fft',]