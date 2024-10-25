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

from __future__ import annotations

import importlib.util
from contextlib import ContextDecorator

import numpy as np

from ._base import Unit, Quantity
from ._unit_common import radian

matplotlib_installed = importlib.util.find_spec('matplotlib') is not None

if matplotlib_installed:
  from matplotlib import ticker, units


  def rad_fn(
      x,
      pos=None
  ) -> str:
    n = int((x / np.pi) * 2.0 + 0.25)
    if n == 0:
      return "0"
    elif n == 1:
      return "π/2"
    elif n == 2:
      return "π"
    elif n % 2 == 0:
      return f"{n // 2}π"
    else:
      return f"{n}π/2"


  class MplQuantityConverter(units.ConversionInterface):
    def __init__(self):
      # Keep track of original converter in case the context manager is
      # used in a nested way.
      self._original_converter = {Quantity: units.registry.get(Quantity)}
      units.registry[Quantity] = self

    @staticmethod
    def axisinfo(unit, axis):
      if unit == radian:
        return units.AxisInfo(
          majloc=ticker.MultipleLocator(base=np.pi / 2),
          majfmt=ticker.FuncFormatter(rad_fn),
          label=unit.dispname,
        )
      elif unit is not None:
        return units.AxisInfo(label=unit.dispname)
      return None

    @staticmethod
    def convert(val, unit, axis):
      if isinstance(val, Quantity):
        return val.mantissa
      elif isinstance(val, list) and val and isinstance(val[0], Quantity):
        return [v.mantissa for v in val]
      else:
        return val

    @staticmethod
    def default_units(x, axis):
      if hasattr(x, "unit"):
        return x.unit
      return None

    def __enter__(self):
      return self

    def __exit__(self, type, value, tb):
      if self._original_converter[Quantity] is None:
        del units.registry[Quantity]
      else:
        units.registry[Quantity] = self._original_converter[Quantity]
