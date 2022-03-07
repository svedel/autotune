from fastapi import APIRouter
from starlette.status import HTTP_201_CREATED
from typing import List
from uuid import UUID
from app.db import PublicCreateExperiment, PublicExperiment


router = APIRouter()


@router.post("/new", response_model=PublicExperiment, status_code=HTTP_201_CREATED)
async def create_new_experiment(new_exp: PublicCreateExperiment):
    return new_exp
