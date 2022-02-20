from fastapi import APIRouter
from typing import List
from uuid import UUID
from starlette.status import HTTP_201_CREATED
from app.core.security import get_password_hash
from app.db import User, CreateUser, PublicUser


router = APIRouter()


@router.get("/", response_model=List[PublicUser])
async def get_users():
    users = await User.objects.select_related("item").all()
    return users

@router.get("/{user_id}", response_model=User, response_model_exclude={"hashed_password"})
async def get_user_id(uuid: UUID):
    user_db = await User.objects.get(uuid=uuid)
    return user_db


@router.post("/signup", response_model=PublicUser, status_code=HTTP_201_CREATED)  # response_model=User
async def create_user_signup(user_in: CreateUser):
    """
    Create new user without the need to be logged in
    """

    # check if user exists already
    user = await User.objects.get_or_none(email=user_in.email)
    if user:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists.",
        )

    # create user if the email address is new
    obj_in = user_in.dict()
    obj_in.pop("password")
    obj_in["hashed_password"] = get_password_hash(user_in.password)
    db_obj = User(**obj_in)

    return await db_obj.save()