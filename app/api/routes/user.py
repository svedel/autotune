from fastapi import APIRouter
from typing import List
from uuid import UUID
from starlette.status import HTTP_201_CREATED
from app.core.security import get_password_hash
from app.db import User, CreateUser, PublicUser


router = APIRouter()


#@router.get("/", response_model=List[User]) #, response_model_exclude={"password"}
@router.get("/", response_model=List[PublicUser])
async def get_users():
    users = await User.objects.select_related("item").all()
    return users

# @router.post("/", response_model=PublicUser, status_code=HTTP_201_CREATED)  # , response_model_exclude={"password"}
# async def create_user(user: CreateUser):
#     obj_in = user.dict()
#     obj_in.pop("password")
#     obj_in["hashed_password"] = get_password_hash(user.password)
#     db_obj = User(**obj_in)
#     return await db_obj.save()

@router.get("/{user_id}", response_model=User, response_model_exclude={"hashed_password"})
async def get_user_id(uuid: UUID):
    user_db = await User.objects.get(uuid=uuid)
    return user_db