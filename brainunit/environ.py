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

# -*- coding: utf-8 -*-

from __future__ import annotations

import contextlib
import dataclasses
import functools
import inspect
import os
import re
import threading
from collections import defaultdict
from typing import Any, Callable, Dict, Hashable

__all__ = [
    # functions for environment settings
    'set', 'context', 'get', 'all',
    # functions for getting default behaviors
    'get_compute_mode',
    # constants
    'SI_MODE', 'NON_SI_MODE'
]

SI_MODE: str = 'si'
NON_SI_MODE: str = 'non_si'


@dataclasses.dataclass
class DefaultContext(threading.local):
    # default environment settings
    settings: Dict[Hashable, Any] = dataclasses.field(default_factory=dict)
    # current environment settings
    contexts: defaultdict[Hashable, Any] = dataclasses.field(default_factory=lambda: defaultdict(list))
    # environment functions
    functions: Dict[Hashable, Any] = dataclasses.field(default_factory=dict)

DEFAULT = DefaultContext()
_NOT_PROVIDE = object()


@contextlib.contextmanager
def context(**kwargs):
    r"""
    Context-manager that sets a computing environment for brainunit.

    For instance::

    >>> import brainunit as u
    >>> global_1 = 2 * u.kmh
    >>> global_2 = 0
    >>> def create_a(a):
    ...     return a.mantissa * 2 * u.minute
    >>> with u.environ.context(compute_mode='si'):
    ...     a = create_a([1, 2, 3] * u.minute)  # If input is [1, 2, 3] * u.second, the result would differ
    ...     b = [4, 5, 6] * u.inch
    ...     global_2 = (b / a) / global_1

    """
    if 'compute_mode' in kwargs:
        if kwargs['compute_mode'] == SI_MODE:
            _convert_to_si_quantity(**kwargs)
        else:
            pass

    try:
        for k, v in kwargs.items():

            # update the current environment
            DEFAULT.contexts[k].append(v)

            # restore the environment functions
            if k in DEFAULT.functions:
                DEFAULT.functions[k](v)

        # yield the current all environment information
        yield all()
    finally:

        for k, v in kwargs.items():

            # restore the current environment
            DEFAULT.contexts[k].pop()

            # restore the environment functions
            if k in DEFAULT.functions:
                DEFAULT.functions[k](get(k))


def get(key: str, default: Any = _NOT_PROVIDE, desc: str = None):
    """
    Get one of the default computation environment.

    Returns
    -------
    item: Any
      The default computation environment.
    """
    if key in DEFAULT.contexts:
        if len(DEFAULT.contexts[key]) > 0:
            return DEFAULT.contexts[key][-1]
    if key in DEFAULT.settings:
        return DEFAULT.settings[key]

    if default is _NOT_PROVIDE:
        if desc is not None:
            raise KeyError(
                f"'{key}' is not found in the context. \n"
                f"You can set it by `brainstate.share.context({key}=value)` "
                f"locally or `brainstate.share.set({key}=value)` globally. \n"
                f"Description: {desc}"
            )
        else:
            raise KeyError(
                f"'{key}' is not found in the context. \n"
                f"You can set it by `brainstate.share.context({key}=value)` "
                f"locally or `brainstate.share.set({key}=value)` globally."
            )
    return default


def all() -> dict:
    """
    Get all the current default computation environment.

    Returns
    -------
    r: dict
      The current default computation environment.
    """
    r = dict()
    for k, v in DEFAULT.contexts.items():
        if v:
            r[k] = v[-1]
    for k, v in DEFAULT.settings.items():
        if k not in r:
            r[k] = v
    return r


def get_compute_mode() -> str:
    """
    Get the current compute mode.

    Returns
    -------
    mode: str
      The current compute mode.
    """
    return get('compute_mode')

def set(
    compute_mode: str = None,
    **kwargs
):
    """
    Set the global default computation environment.



    Args:
      compute_mode: str, optional
        The default compute mode. Default is computing in 'si'.
    """
    if compute_mode is not None:
        assert compute_mode in ['si', 'non_si'], f"compute_mode must be 'si' or 'non_si'. Got: {compute_mode}"
        kwargs['compute_mode'] = compute_mode

    # set default environment
    DEFAULT.settings.update(kwargs)

    # update the environment functions
    for k, v in kwargs.items():
        if k in DEFAULT.functions:
            DEFAULT.functions[k](v)

def _convert_to_si_quantity(**kwargs):
    """
                Convert all the local variables in SI units.

                Traverses the local variables in the calling scope and converts all `Quantity`
                instances (including those nested in lists, tuples, or dictionaries) to their SI unit equivalents.
                The conversion is performed by calling the `factorless()` method on each `Quantity` instance,
                which convert the unit and returns the quantities in SI units.
                """
    set(compute_mode=kwargs['compute_mode'])
    from ._base import Quantity, Unit
    frame = inspect.currentframe().f_back.f_back.f_back
    original = {k: v for k, v in frame.f_locals.items()
                if isinstance(v, (Quantity, Unit))}

    try:
        # Convert to SI
        for k, v in original.items():
            frame.f_locals[k] = v.factorless()
        yield
    finally:
        # Restore original values
        for k, v in original.items():
            frame.f_locals[k] = v

set(compute_mode=NON_SI_MODE)