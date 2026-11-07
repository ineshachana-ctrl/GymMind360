"""End-to-end smoke test: every page renders without raising an exception."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted(p.name for p in (ROOT / "views").glob("*.py"))


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(app, page):
    app.switch_page(f"views/{page}")
    app.run()
    assert not app.exception, "\n".join(e.value for e in app.exception)
