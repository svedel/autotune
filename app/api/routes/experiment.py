from fastapi import APIRouter, Security, HTTPException
from starlette.status import HTTP_201_CREATED, HTTP_200_OK
from jose import JWTError, jwt, ExpiredSignatureError
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import ValidationError
from uuid import UUID
from typing import List

from app.db import PublicCreateExperiment, PublicExperiment, User, Experiment, PublicExperimentAsk, PublicExperimentBase
from app.core.auth import bearer_scheme
from app.core.config import settings
from app.api.helpers import ExperimentOperations


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

        # parse the provided experiment and cast for Experiment class in db
        exp = ExperimentOperations.parse_new_experiment(new_exp=new_exp, user=user)

        await exp.save()

        # retrieve stored result to return
        new_exp_public = await ExperimentOperations.public_experiment(exp_uuid=exp.exp_uuid)

        return new_exp_public

    except (JWTError, ValidationError):
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    except ExpiredSignatureError:
        raise HTTPException(status_code=403, detail="Token expired")


@router.get("/ask/{exp_uuid}", response_model=PublicExperimentAsk, status_code=HTTP_200_OK)
async def experiment_ask(exp_uuid: UUID, credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    '''
    endpoint to retrieve the covariates for which the algorithm believes the response will generate the most new
    knowledge wrt finding the optimum
    :param exp_uuid (UUID): unique identifier for experiment
    :param credentials:
    :return:
    '''

    try:
        # decode
        payload = jwt.decode(credentials.credentials, settings.JWT_SECRET, algorithms=[settings.ALGORITHM])

        # check user
        user = await User.objects.filter(id=int(payload.get("sub"))).first()
        if not payload["type"] == "access_token":
            raise HTTPException(status_code=401, detail="Invalid token")
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")

        # check user has access to experiment
        exp = await Experiment.objects.filter(exp_uuid=exp_uuid).first()
        await exp.user.load()  # load the user for this exp from the db
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")
        if not exp.user.uuid == user.uuid:
            raise HTTPException(status_code=403, detail="Forbidden. User does not have access to this experiment")

        # determine covars for next experiment via TuneSession's ask-method, update experiment and return
        next_covars = await ExperimentOperations.ask_next_datapoint(exp)

        return next_covars

    except (JWTError, ValidationError):
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    except ExpiredSignatureError:
        raise HTTPException(status_code=403, detail="Token expired")


@router.get("/all", response_model=List[PublicExperimentBase], response_model_exclude={"user_uuid"}, status_code=HTTP_200_OK)
async def experiment_all(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    '''
    endpoint to post all experiments by user with provided credentials, ordered by last update date
    '''

    try:
        # decode
        payload = jwt.decode(credentials.credentials, settings.JWT_SECRET, algorithms=[settings.ALGORITHM])

        # check user
        user = await User.objects.filter(id=int(payload.get("sub"))).first()
        if not payload["type"] == "access_token":
            raise HTTPException(status_code=401, detail="Invalid token")
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")

        # find all experiments where user is JWT user
        #experiments = await Experiment.objects.filter(user__uuid=user.uuid).all()
        experiments = await Experiment.objects.filter(user__uuid=user.uuid).order_by(Experiment.time_updated.desc()).all()

        # sort experiment based on last updated time

        print(experiments)

        return experiments

    except (JWTError, ValidationError):
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    except ExpiredSignatureError:
        raise HTTPException(status_code=403, detail="Token expired")


# ednpoint to report results

#@router.get("/list", response_model=PublicExperimentAsk, status_code=HTTP_200_OK)
# create an endpoint to see all experiments for a particular user
