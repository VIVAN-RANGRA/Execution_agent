"""Smoke tests for the Streamlit dashboard module."""
import runpy
import sys
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parent.parent
DASHBOARD_APP = BASE_DIR / "dashboard" / "app.py"


class _FakeBlock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeStreamlit:
    def __init__(self):
        self.sidebar = _FakeBlock()

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cache_resource(self, fn=None, **_kwargs):
        if fn is None:
            return lambda wrapped: wrapped
        return fn

    def cache_data(self, fn=None, **_kwargs):
        if fn is None:
            return lambda wrapped: wrapped
        return fn

    def cache_data(self, fn=None, **_kwargs):
        if fn is None:
            return lambda wrapped: wrapped
        return fn

    def tabs(self, labels):
        return [_FakeBlock() for _ in labels]

    def columns(self, count):
        return [_FakeBlock() for _ in range(count)]

    def spinner(self, _label):
        return _FakeBlock()

    def multiselect(self, _label, options, default=None, **_kwargs):
        return list(default or options[:1])

    def slider(self, _label, **kwargs):
        return kwargs.get("value")

    def selectbox(self, _label, options, index=0, **_kwargs):
        return options[index]

    def button(self, *_args, **_kwargs):
        return False

    def number_input(self, _label, value=0, **_kwargs):
        return value

    def set_page_config(self, **_kwargs):
        return None

    def title(self, *_args, **_kwargs):
        return None

    def markdown(self, *_args, **_kwargs):
        return None

    def header(self, *_args, **_kwargs):
        return None

    def subheader(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None

    def success(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def plotly_chart(self, *_args, **_kwargs):
        return None


def test_dashboard_import_smoke(monkeypatch):
    pytest.importorskip("plotly")
    fake_streamlit = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    runpy.run_path(str(DASHBOARD_APP), run_name="__main__")
