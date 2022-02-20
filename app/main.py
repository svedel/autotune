from fastapi import FastAPI
from app.api.routes import router as api_router
from app.db import database, User, Item
from app.core.config import settings
from app.core.security import get_password_hash


tags_metadata = [
    {
        "name": "User",
        "description": "Operations for existing users.",
    },
    {
        "name": "Item",
        "description": "Operations for items.",
    },
    {
        "name": "Auth",
        "description": "Operations for signing up and user auth (user creation (sign-up) and sign-in).",
    }
]


app = FastAPI(title=settings.project_name, version=settings.project_version, openapi_tags=tags_metadata)
app.include_router(api_router)


@app.get("/")
async def read_root():
    return await User.objects.all()

@app.on_event("startup")
async def startup():
    if not database.is_connected:
        await database.connect()

    hat = await Item.objects.create(name="hat")
    await User.objects.get_or_create(email="test@test.com", item=hat, hashed_password=get_password_hash("CHANGEME"))
    await User.objects.get_or_create(email="me@somewhere.com", hashed_password=get_password_hash("CHANGEME"))

@app.on_event("shutdown")
async def shutdown():
    if database.is_connected:
        await database.disconnect()
