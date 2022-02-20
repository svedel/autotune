import ormar
import uuid
from datetime import datetime

from app.db.core import BaseMeta


class Item(ormar.Model):
    class Meta(BaseMeta):
        tablename: str = "items"

    id: int = ormar.Integer(primary_key=True, autoincrement=True)
    uuid: str = ormar.UUID(uuid_format="string", default=uuid.uuid4, index=True)
    time_created: datetime = ormar.DateTime(default=datetime.utcnow)
    name: str = ormar.String(max_length=254, unique=True, nullable=False)
