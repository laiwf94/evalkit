from alembic import context
from sqlalchemy import Engine, text

from evalkit.db.models import Base

# Arbitrary constant: every evalkit process uses the same lock id, so replicas
# starting together migrate one at a time instead of racing on CREATE TABLE.
MIGRATION_LOCK_ID = 7_301_482_911

engine: Engine | None = context.config.attributes.get("engine")
if engine is None:
    raise RuntimeError("run migrations through evalkit.db.upgrade, which supplies the engine")

with engine.connect() as connection:
    postgres = connection.dialect.name == "postgresql"
    if postgres:
        connection.execute(text("SELECT pg_advisory_lock(:id)"), {"id": MIGRATION_LOCK_ID})
        connection.commit()
    try:
        context.configure(
            connection=connection,
            target_metadata=Base.metadata,
            render_as_batch=connection.dialect.name == "sqlite",
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    finally:
        if postgres:
            connection.execute(text("SELECT pg_advisory_unlock(:id)"), {"id": MIGRATION_LOCK_ID})
            connection.commit()
