from typing import Optional, MutableMapping, List, Union
from datetime import datetime, timedelta

from fastapi.security import OAuth2PasswordBearer, HTTPBearer
from jose import jwt

from app.db import User
from app.core.config import settings
from app.core.security import verify_password


JWTPayloadMapping = MutableMapping[str, Union[datetime, bool, str, List[str], List[int]]]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

bearer_scheme = HTTPBearer()


async def get_user_by_email(email: str) -> Optional[User]:
    return await User.objects.filter(email=email).first()  # User.objects.get_or_none(email=email)


async def authenticate(email: str, password: str) -> Optional[User]:
    """
    authenticate user
    """

    user = await get_user_by_email(email=email)

    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(*, sub: str) -> str:
    return _create_token(
        token_type="access_token",
        lifetime=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        sub=sub,
    )


def _create_token(token_type: str, lifetime: timedelta, sub: str) -> str:
    """
    create access token with lifetime
    """

    # initializing
    payload = {}
    expire = datetime.utcnow() + lifetime
    payload["type"] = token_type

    # https://datatracker.ietf.org/doc/html/rfc7519#section-4.1.3
    # The "exp" (expiration time) claim identifies the expiration time on
    # or after which the JWT MUST NOT be accepted for processing
    payload["exp"] = expire

    # The "iat" (issued at) claim identifies the time at which the
    # JWT was issued.
    payload["iat"] = datetime.utcnow()

    # The "sub" (subject) claim identifies the principal that is the
    # subject of the JWT
    payload["sub"] = str(sub)

    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.ALGORITHM)


def create_refresh_token(*, sub: str) -> str:
    return _create_token(
        token_type="refresh_token",
        lifetime=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        sub=sub,
    )


def refresh_token(*, refresh_token) -> str:
    payload = jwt.decode(refresh_token, settings.JWT_SECRET, algorithms=[settings.ALGORITHM])
    if (payload["type"] == "refresh_token"):
        sub = payload["sub"]
