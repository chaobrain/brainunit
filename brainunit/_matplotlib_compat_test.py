import pytest

import brainunit as u

try:
  import matplotlib.pyplot as plt
except ImportError:
  pytest.skip("matplotlib is not installed", allow_module_level=True)


def test_quantity_support():
  with u.matplotlib_support_quantity():
    plt.figure()
    plt.plot([1, 2, 3] * u.meter)
    plt.show()

    plt.cla()
    plt.plot([101, 125, 150] * u.cmeter)
    plt.show()

    plt.cla()
    plt.plot([101, 125, 150] * u.ms, [101, 125, 150] * u.cmeter)
    plt.plot([0.1, 0.15, 0.2] * u.second, [111, 135, 160] * u.cmeter)
    plt.show()

    plt.cla()
    plt.plot([101, 125, 150] * u.ms, [101, 125, 150] * u.cmeter)
    plt.plot([0.1, 0.15, 0.2] * u.second, [111, 135, 160] * u.cmeter)
    plt.plot([0.1, 0.15, 0.2] * u.second, [131, 155, 180] * u.mA)
    plt.show()

  with pytest.raises(TypeError):
    plt.figure()
    plt.plot([1, 2, 3] * u.meter)
    plt.show()

    plt.cla()
    plt.plot([101, 125, 150] * u.cmeter, [1, 2, 3] * u.kgram)
    plt.show()
