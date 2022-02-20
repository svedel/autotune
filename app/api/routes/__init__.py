from fastapi import APIRouter
from app.api.routes.user import router as user_router
from app.api.routes.item import router as item_router
from app.api.routes.auth import router as auth_router


router = APIRouter()


router.include_router(user_router, prefix="/user", tags=["User"])
router.include_router(item_router, prefix="/item", tags=["Item"])
router.include_router(auth_router, prefix="/auth", tags=["Auth"])
