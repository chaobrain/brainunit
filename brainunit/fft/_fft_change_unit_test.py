import itertools

import jax.numpy as jnp
import jax.numpy.fft as jnpfft
import numpy as np
from absl.testing import parameterized

import brainunit as u
import brainunit.fft as ufft
from brainunit import meter, second, volt
from brainunit._base import assert_quantity

class TestFftChangeUnit(parameterized.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFftChangeUnit, self).__init__(*args, **kwargs)

        print()

    def test_time_freq_map(self):
        from brainunit.fft._fft_change_unit import _time_freq_map
        for key, value in _time_freq_map.items():
            # print(key.scale, value.scale)
            assert key.scale == -value.scale
