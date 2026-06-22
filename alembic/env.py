from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from kodiak.config.settings import get_settings
from kodiak.db.base import Base
import kodiak.db.models.user  # noqa: F401
import kodiak.db.models.repository  # noqa: F401
import kodiak.db.models.task  # noqa: F401
import kodiak.db.models.agent_run  # noqa: F401
import kodiak.db.models.memory  # noqa: F401
import kodiak.db.models.pull_request  # noqa: F401
import kodiak.db.models.approval  # noqa: F401
import kodiak.db.models.plugin  # noqa: F401
import kodiak.db.models.learning  # noqa: F401
import kodiak.db.models.audit  # noqa: F401

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(url=get_settings().database_url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_settings().database_url.replace("+asyncpg", "")
    connectable = engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
