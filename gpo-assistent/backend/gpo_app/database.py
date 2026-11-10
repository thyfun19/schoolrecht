"""PostgreSQL configuration and connection management for the GPO source registry."""

import os
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.engine import make_url

BASE = Path(__file__).resolve().parents[1]


def database_url():
    value = os.environ.get("GPO_REGISTRY_DATABASE_URL")
    if not value:
        path = BASE / "data/registry/database-url.txt"
        if not path.exists():
            raise ValueError(
                "GPO-bronregister niet geconfigureerd. Start eerst local-registry.ps1."
            )
        value = path.read_text(encoding="utf-8").strip()
    if make_url(value).get_backend_name() != "postgresql":
        raise ValueError(
            "Dit bronregister vereist PostgreSQL; geen stille SQLite-vervanging."
        )
    return value


def create_engine(url=None):
    return sa.create_engine(
        url or database_url(),
        pool_pre_ping=True,
        hide_parameters=True,
        connect_args={"connect_timeout": 5},
    )
