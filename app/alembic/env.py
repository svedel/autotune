from logging.config import fileConfig
from sqlalchemy import create_engine

import sys, os

# add app folder to system path
#sys.path.append(os.getcwd())
#myPath = os.path.dirname(os.path.abspath(__file__))
#sys.path.insert(0, myPath + '/../../')
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)

print(parent_dir)

from app.core.config import settings
from app.db.core import metadata
from alembic import context

# this is the Alembic config object which provides access to the values in the .ini file
config = context.config

# interpret the config file for logging purposes
fileConfig(config.config_file_name)

# note how it's 'raw' metadata not the one attached to Base as there is no Base
target_metadata = metadata
URL = settings.db_url


def run_migrations_offline():

    context.configure(
        url=URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        user_module_prefix='sa.'
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = create_engine(URL)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            user_module_prefix='sa.'
        )

    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
