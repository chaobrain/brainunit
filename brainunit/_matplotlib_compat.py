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
from typing import List

import numpy as np
from brainstate.typing import ArrayLike
from numpy import ma

from ._base import Quantity, fail_for_dimension_mismatch, UNITLESS, UnitMismatchError

matplotlib_installed = importlib.util.find_spec('matplotlib') is not None

__all__ = [
  "set_axis_unit",
]

if matplotlib_installed:
  from matplotlib import ticker, units, axis, pyplot as plt
  from matplotlib.lines import Line2D

  # setattr(plt, "_plot", plt.plot)
  #
  #
  # def plotq(
  #     *args: float | ArrayLike | str,
  #     scalex: bool = True,
  #     scaley: bool = True,
  #     data=None,
  #     **kwargs,
  # ) -> List[Line2D]:
  #   # args to Quantity
  #   args = [arg if isinstance(arg, Quantity) else Quantity(arg) for arg in args]
  #   return plt._plot(*args, scalex=scalex, scaley=scaley, data=data, **kwargs)
  #
  #
  # setattr(plt, "plot", plotq)


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

    @staticmethod
    def axisinfo(unit, axis):
      if unit == UNITLESS:
        return units.AxisInfo()
      elif unit is not None:
        return units.AxisInfo(label=unit.dispname)
      return None

    @staticmethod
    def convert(val, unit, axis):
      if isinstance(val, Quantity):
        # check dimension
        fail_for_dimension_mismatch(val.unit, unit)
        # check unit
        if val.unit != unit:
          # scale to target unit
          return val.to(unit).mantissa
        return val.mantissa
      elif isinstance(val, list) and val and isinstance(val[0], Quantity):
        fail_for_dimension_mismatch(val[0].unit, unit)
        return [v.to(unit).mantissa if v.unit != unit else v.mantissa for v in val]
      else:
        if unit is UNITLESS:
          return val
        else:
          raise UnitMismatchError(UNITLESS, unit)

    @staticmethod
    def default_units(x, axis):
      if hasattr(x, "unit"):
        return x.unit
      return UNITLESS

  # check decimal

  class DecimalConverter(units.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):
      # check value unit is UNITLESS
      if isinstance(value, Quantity):
        if value.unit is UNITLESS:
          raise UnitMismatchError(value.unit, UNITLESS)
        return value.mantissa

      if isinstance(value, units.Decimal):
        result = float(value)
      # value is Iterable[Decimal]
      elif isinstance(value, ma.MaskedArray):
        result = ma.asarray(value, dtype=float)
      else:
        result = np.asarray(value, dtype=float)
      return Quantity(result)

    @staticmethod
    def default_units(x, axis):
      if hasattr(x, "unit"):
        return x.unit
      return UNITLESS


  def convert_units(self, x):
    # If x is natively supported by Matplotlib, doesn't need converting
    if units._is_natively_supported(x):
      return x

    if self.converter is None:
      self.converter = units.registry.get_converter(x)

    if self.converter is None:
      return x
    try:
      ret = self.converter.convert(x, self.units, self)
    except Exception as e:
      raise units.ConversionError('Failed to convert value(s) to axis '
                                   f'units: {x!r}') from e
    return ret

  axis.Axis.convert_units = convert_units
  units.registry[units.Decimal] = DecimalConverter()
  units.registry[np.number] = DecimalConverter()
  units.registry[Quantity] = MplQuantityConverter()


  def set_axis_unit(axis_index, target_unit, ax=None, precision=None):
    """
    Set the scale of the specified axis to the target unit.

    Parameters:
    - axis_index: index of the axis (0 for x-axis, 1 for y-axis)
    - target_unit: target unit to convert the axis scale to
    - ax: matplotlib axis object
    - auto_precision: automatically adjust the precision of the axis labels
    """
    if ax is None:
      ax = plt.gca()

    if axis_index == 0:
      axis = ax.xaxis
    elif axis_index == 1:
      axis = ax.yaxis
    else:
      raise ValueError("Invalid axis index. Use 0 for x-axis or 1 for y-axis.")

    # Get the current unit of the axis
    current_unit = axis.units

    # Set the formatter and locator
    if precision is not None:
      formatter = lambda x, _: f"{((x * current_unit).to(target_unit)).mantissa:.{precision}f}"
    else:
      formatter = lambda x, _: f"{((x * current_unit).to(target_unit)).mantissa}"

    axis.set_major_formatter(ticker.FuncFormatter(formatter))
    axis.set_major_locator(ticker.AutoLocator())

    # Update label
    if axis_index == 0:
      ax.set_xlabel(f"{ax.get_xlabel()} ({target_unit})")
    else:
      ax.set_ylabel(f"{ax.get_ylabel()} ({target_unit})")
else:
  set_axis_unit = None
