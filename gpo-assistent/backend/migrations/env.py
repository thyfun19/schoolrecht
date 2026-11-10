from alembic import context
from gpo_app.database import create_engine
from gpo_app.models import metadata


def configure(connection=None):
    context.configure(
        connection=connection,
        target_metadata=metadata,
        version_table="gpo_alembic_version",
        compare_type=True,
        **(
            {"dialect_name": "postgresql", "literal_binds": True}
            if connection is None
            else {}
        ),
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    configure()
else:
    engine = create_engine(context.config.attributes.get("database_url"))
    with engine.connect() as connection:
        configure(connection)
    engine.dispose()
