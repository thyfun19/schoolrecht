"""Integration tests use newly created PostgreSQL databases, never the real registry."""

import re
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import make_url

from gpo_app.database import create_engine, database_url
from gpo_app.ingestion import prepare_existing
from gpo_app import storage

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]


@pytest.fixture(scope="session")
def prepared():
    return prepare_existing(
        BASE.parent / "sources/catalogus.json",
        ROOT / "onderzoek_gpo/data/pdf",
        ROOT / "onderzoek_gpo/data/pages.json",
        BASE / "data/e5/corpus.json",
    )


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DEFAULT_STORAGE_ROOT", tmp_path / "managed-pdfs")
    name = "gpo_registry_test_" + uuid.uuid4().hex
    assert re.fullmatch(r"gpo_registry_test_[a-f0-9]{32}", name)
    url = make_url(database_url())
    admin = create_engine(url.set(database="postgres"))
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as con:
        con.exec_driver_sql(f'CREATE DATABASE "{name}"')
    test_url = url.set(database=name)
    result = create_engine(test_url)
    try:
        config = Config(str(BASE / "alembic-registry.ini"))
        config.attributes["database_url"] = test_url
        command.upgrade(config, "head")
        yield result
    finally:
        result.dispose()
        # Drop only the random database created above, not a configured user database.
        assert re.fullmatch(r"gpo_registry_test_[a-f0-9]{32}", name)
        with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as con:
            con.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()
