"""Initialize credentials for this new local development database only."""

import secrets
from pathlib import Path
from sqlalchemy.engine import URL

BASE = Path(__file__).resolve().parents[1]


def initialize():
    directory = (BASE / "data/registry").resolve()
    if not directory.is_relative_to(BASE):
        raise ValueError("Configuratiepad ligt buiten backend.")
    directory.mkdir(parents=True, exist_ok=True)
    password_path = directory / "postgres-password.txt"
    url_path = directory / "database-url.txt"
    if not password_path.exists():
        if url_path.exists():
            raise ValueError(
                "Onvolledige configuratie; bestaande URL niet overschrijven."
            )
        with password_path.open("x", encoding="utf-8") as stream:
            stream.write(secrets.token_hex(32))
    password = password_path.read_text(encoding="utf-8").strip()
    url = URL.create(
        "postgresql+psycopg",
        username="gpo_dev",
        password=password,
        host="127.0.0.1",
        port=55432,
        database="gpo_registry",
    ).render_as_string(hide_password=False)
    if url_path.exists():
        if not secrets.compare_digest(
            url_path.read_text(encoding="utf-8").strip(), url
        ):
            raise ValueError(
                "Bestaande lokale configuratie verschilt; niet overschreven."
            )
    else:
        with url_path.open("x", encoding="utf-8") as stream:
            stream.write(url)
    print("Lokale GPO-databaseconfiguratie gereed; credentials niet getoond.")


if __name__ == "__main__":
    initialize()
