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

__all__ = [
    # math funcs only accept unitless (unary)
    'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh',
    'collapse', 'cumlogsumexp',
    'bessel_i0e', 'bessel_i1e', 'digamma', 'lgamma', 'igamma', 'erf', 'erfc',
    'erf_inv', 'logistic',


    # math funcs only accept unitless (binary)
    'atan2', 'polygamma', 'igammac', 'igamma_grad_a', 'random_gamma_grad',
    'zeta',

    # math funcs only accept unitless (n-ary)
    'betainc', 'betainc_gradx', 'betainc_grad_not_implemented', 'igamma_gradx',
    'igamma_grada', 'igammac_gradx', 'igammac_grada', 'polygamma_gradm', 'polygamma_gradx',

    # Elementwise bit operations (unary)

    # Elementwise bit operations (binary)
    'shift_left', 'shift_right_arithmetic', 'shift_right_logical',
]

# math funcs only accept unitless (unary)
def acos(x): pass
def acosh(x): pass
def asin(x): pass
def asinh(x): pass
def atan(x): pass
def atanh(x): pass
def collapse(x): pass
def cumlogsumexp(x): pass
def bessel_i0e(x): pass
def bessel_i1e(x): pass
def digamma(x): pass
def lgamma(x): pass
def igamma(x): pass
def erf(x): pass
def erfc(x): pass
def erf_inv(x): pass
def logistic(x): pass

# math funcs only accept unitless (binary)
def atan2(x, y): pass
def polygamma(x, y): pass
def igammac(x, y): pass
def igamma_grad_a(x, y): pass
def random_gamma_grad(x, y): pass
def zeta(x, y): pass

# math funcs only accept unitless (n-ary)
def betainc(x, y, z): pass
def betainc_gradx(x, y, z): pass
def betainc_grad_not_implemented(x, y, z): pass
def igamma_gradx(x, y, z): pass
def igamma_grada(x, y, z): pass
def igammac_gradx(x, y, z): pass
def igammac_grada(x, y, z): pass
def polygamma_gradm(x, y, z): pass
def polygamma_gradx(x, y, z): pass


# Elementwise bit operations (binary)
def shift_left(x, y): pass
def shift_right_arithmetic(x, y): pass
def shift_right_logical(x, y): pass