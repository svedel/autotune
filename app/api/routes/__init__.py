from fastapi import APIRouter
from app.api.routes.user import router as user_router
from app.api.routes.auth import router as auth_router
from app.api.routes.experiment import router as exp_router


router = APIRouter()


router.include_router(user_router, prefix="/user", tags=["User"])
router.include_router(auth_router, prefix="/auth", tags=["Authentication and authorization"])
router.include_router(exp_router, prefix="/experiment", tags=["Experiment"])
