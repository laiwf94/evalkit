from alembic import context
from sqlalchemy import Engine

from evalkit.db.models import Base

engine: Engine | None = context.config.attributes.get("engine")
if engine is None:
    raise RuntimeError("run migrations through evalkit.db.upgrade, which supplies the engine")

with engine.connect() as connection:
    context.configure(
        connection=connection,
        target_metadata=Base.metadata,
        render_as_batch=connection.dialect.name == "sqlite",
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()
