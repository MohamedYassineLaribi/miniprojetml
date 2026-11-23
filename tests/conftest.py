import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # permet `from main import app`

from main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    return TestClient(app)
