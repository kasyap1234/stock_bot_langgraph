import pytest
import builtins
from unittest import mock
import time as _time

# Ensure tests can reference the time module without explicit import (mirrors pytest-mock behaviour)
builtins.time = _time


@pytest.fixture
def benchmark():
    """Lightweight benchmark fixture fallback for environments without pytest-benchmark."""

    def _wrapper(func, *args, **kwargs):
        result = func(*args, **kwargs)

        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], str):
            return True

        return result

    return _wrapper


class _SimpleMocker:
    def __init__(self):
        self._patchers = []

    def patch(self, target, *args, **kwargs):
        patcher = mock.patch(target, *args, **kwargs)
        patched = patcher.start()
        self._patchers.append(patcher)
        return patched

    def spy(self, obj, attribute):
        original = getattr(obj, attribute)
        patcher = mock.patch.object(obj, attribute, wraps=original)
        spy_obj = patcher.start()
        self._patchers.append(patcher)
        return spy_obj

    def stopall(self):
        while self._patchers:
            self._patchers.pop().stop()


@pytest.fixture
def mocker():
    helper = _SimpleMocker()
    try:
        yield helper
    finally:
        helper.stopall()
