from __future__ import annotations

import pytest

from alframework.ml_interfaces.excited_state_hippynn_interface import DataDumper


class _FakePlotter:
    value = 7


def test_data_dumper_getattr_raises_attribute_error_before_plotter_restore():
    dumper = DataDumper.__new__(DataDumper)

    with pytest.raises(AttributeError):
        dumper.any_missing_attribute
    with pytest.raises(AttributeError):
        dumper._plotter


def test_data_dumper_getattr_delegates_after_plotter_restore():
    dumper = DataDumper.__new__(DataDumper)
    dumper._plotter = _FakePlotter()

    assert dumper.value == 7
    with pytest.raises(AttributeError):
        dumper.missing_from_plotter
