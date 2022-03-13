from fastapi import APIRouter, Security, HTTPException
from starlette.status import HTTP_201_CREATED
from jose import JWTError, jwt, ExpiredSignatureError
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import ValidationError
from typing import List
from uuid import UUID

from app.db import PublicCreateExperiment, PublicExperiment, User
from app.core.auth import bearer_scheme
from app.core.config import settings


router = APIRouter()


@router.post("/new", response_model=PublicExperiment, status_code=HTTP_201_CREATED)
async def create_new_experiment(
        new_exp: PublicCreateExperiment,
        credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)
):
    '''
    secured API endpoint to create new experiment. User must provide valid JWT (token) to set up new experiment via
    this endpoint.

    using PublicExperiment data model to provide exp details
    '''

    try:
        # decode
        payload = jwt.decode(credentials.credentials, settings.JWT_SECRET, algorithms=[settings.ALGORITHM])

        # check user
        user = await User.objects.filter(id=int(payload.get("sub"))).first()
        # user = await get_user_by_email(email=payload.get("sub"))
        if not payload["type"] == "access_token":
            raise HTTPException(status_code=401, detail="Invalid token")
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")

        return new_exp

    except (JWTError, ValidationError):
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    except ExpiredSignatureError:
        raise HTTPException(status_code=403, detail="Token expired")
