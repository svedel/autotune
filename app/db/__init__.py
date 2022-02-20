import sqlalchemy

from app.db.user import User, CreateUser, PublicUser
from app.db.item import Item
from app.db.core import metadata, database
from app.db.token import Token, TokenData

from app.core.config import settings


engine = sqlalchemy.create_engine(settings.db_url, pool_timeout=60)
metadata.create_all(engine)
