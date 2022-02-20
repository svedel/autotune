import databases
import ormar
import sqlalchemy

from app.core.config import settings


database = databases.Database(settings.db_url)
metadata = sqlalchemy.MetaData()


class BaseMeta(ormar.ModelMeta):
    metadata = metadata
    database = database