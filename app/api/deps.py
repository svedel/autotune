from typing import Optional
from pydantic import BaseModel, ValidationError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic
from jose import jwt, JWTError

from app.core.auth import oauth2_scheme
from app.core.config import settings
from app.db import User, TokenData


bearer_security = HTTPBasic()  #scheme_name="Authorization"


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    store authorized, signed-in user, allow for retrieval
    """

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.ALGORITHM],
            options={"verify_aud": False},
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = await User.objects.filter(id=token_data.username).first()
    if user is None:
        raise credentials_exception
    return user


async def validate_token(http_authorization_credentials=Depends(bearer_security)):
    """
    get the JWT and decode the token for username and password
    """

    print(http_authorization_credentials)

    try:
        # decode
        payload = jwt.decode(http_authorization_credentials.credentials, settings.JWT_SECRET,
                             algorithm=settings.ALGORITHM)

        # check user
        user_db = await User.objects.filter(id=int(payload.get("sub"))).first()
        if not user_db:
            raise HTTPException(status_code=401, detail="Invalid token")
        if not payload.get("exp") < datetime.utcnow():
            raise HTTPException(status_code=403, detail="Token expired")

        return {
            "username": user.email,
            "uuid": user.uuid,
            "access_token": create_access_token(sub=user.id),
            "token_type": "bearer",
        }

    except (JWTError, ValidationError):
        raise HTTPException(status_code=403, detail="Could not validate credentials")

