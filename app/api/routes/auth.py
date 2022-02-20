from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Request, Security
from jose import JWTError, jwt, ExpiredSignatureError
from pydantic import ValidationError
from starlette.status import HTTP_201_CREATED, HTTP_400_BAD_REQUEST
from fastapi.security import OAuth2PasswordRequestForm, HTTPBasicCredentials, HTTPAuthorizationCredentials
from app.db import User, PublicUser, CreateUser, Token, TokenData
from app.core.auth import authenticate, create_access_token, bearer_scheme, refresh_token, get_user_by_email
from app.core.config import settings
from app.core.security import get_password_hash
from app.api import deps


router = APIRouter()


@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    get the JWT for a user with data from OAuth2 request form body
    """

    user = await authenticate(email=form_data.username, password=form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    return {
        "access_token": create_access_token(sub=user.id),
        "token_type": "bearer",
    }


@router.get("/me", response_model=PublicUser)
async def read_user_me(current_user: User = Depends(deps.get_current_user)):
    """
    gets the current logged in user
    """

    user = current_user
    return user

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

@router.post("/token", response_model=Token, status_code=HTTP_201_CREATED)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate(email=form_data.username, password=form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    return {
        "access_token": create_access_token(sub=user.id),
        "token_type": "bearer",
    }

#@router.get("/refresh_token", response_model=Token, status_code=HTTP_201_CREATED)
#async def get_refresh_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):


@router.get("/get-me")
async def confirm_user_me(credentials: HTTPBasicCredentials = Depends(deps.bearer_security)):
    """
    get the JWT for a user with data from OAuth2 request via HTTP header
    """

    user = await authenticate(email=credentials.username, password=credentials.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    return {
        "username": user.email,
        "uuid": user.uuid,
        "access_token": create_access_token(sub=user.id),
        "token_type": "bearer",
    }


@router.post("/header-me")
async def confirm_user_header_me(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    """
    get the JWT and decode the token for username and password

    :param credentials: HTTPAuthorizationCredentials object, contains token in attribute 'credentials'
    """

    try:
        # decode
        payload = jwt.decode(credentials.credentials, settings.JWT_SECRET, algorithms=[settings.ALGORITHM])

        # check user
        user = await User.objects.filter(id=int(payload.get("sub"))).first()
        #user = await get_user_by_email(email=payload.get("sub"))
        if not payload["type"] == "access_token":
            raise HTTPException(status_code=401, detail="Invalid token")
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")

        return {
            "username": user.email,
            "uuid": user.uuid,
            "access_token": create_access_token(sub=user.id),
            "token_type": "bearer",
        }

    except (JWTError, ValidationError):
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    except ExpiredSignatureError:
        raise HTTPException(status_code=403, detail="Token expired")




