from app.db.user import User
from app.core.security import get_password_hash


# create first users
async def init_db():
    await User.objects.get_or_create(email="test@test.com", hashed_password=get_password_hash("CHANGEME"))
    await User.objects.get_or_create(email="me@somewhere.com", hashed_password=get_password_hash("CHANGEME"))


if __name__ == "__main__":
    init_db()
