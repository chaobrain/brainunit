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

import unittest

import pytest
import brainunit as u

class TestEnviron(unittest.TestCase):
    def test_compute_mode(self):
        global_1 = 2 * u.kmh
        global_2 = 0

        def create_a(a):
            return a.mantissa * 2 * u.minute

        with u.environ.context(compute_mode='si'):
            a = create_a([1, 2, 3] * u.minute)  # If input is [1, 2, 3] * u.second, the result would differ
            b = [4, 5, 6] * u.inch
            global_2 = (b / a) / global_1

            assert a.unit.factor == 1.
            assert b.unit.factor == 1.
            # TODO: need to fix compound standard units
            # assert global_1.unit.factor == 1.