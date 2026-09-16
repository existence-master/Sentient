from __future__ import annotations

import pytest

from sentient.app import SentientApp
from tests.conftest import FakeProvider


@pytest.fixture
async def app(config, isolated_home):
    llm = FakeProvider()
    a = SentientApp(config, llm=llm, db_path=isolated_home / "mem.db", enable_background=False)
    await a.start()
    a.fake = llm
    yield a
    await a.stop()
