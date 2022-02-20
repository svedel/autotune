import ormar
import uuid
from pydantic import EmailStr
from typing import Optional
from datetime import datetime

from app.db.core import BaseMeta
from app.db.item import Item


class User(ormar.Model):
    class Meta(BaseMeta):
        tablename: str = "users"

    id: int = ormar.Integer(primary_key=True, autoincrement=True)
    uuid: str = ormar.UUID(uuid_format="string", default=uuid.uuid4, index=True)
    email: EmailStr = ormar.String(max_length=128, unique=True, nullable=False, index=True)
    hashed_password: str = ormar.String(max_length=255, nullable=False)
    time_created: datetime = ormar.DateTime(default=datetime.utcnow, nullable=False)
    active: bool = ormar.Boolean(default=True, nullable=False)
    item: Optional[Item] = ormar.ForeignKey(Item, nullable=True)


# create pydantic model for User without password
SimpleUser = User.get_pydantic(include={"email"})
class CreateUser(SimpleUser):
    password: str


PublicUser = User.get_pydantic(exclude={"hashed_password", "item"})
