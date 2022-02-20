from fastapi import APIRouter
from starlette.status import HTTP_201_CREATED
from typing import List
from uuid import UUID
from app.db import Item


router = APIRouter()


@router.post("/", response_model=Item, status_code=HTTP_201_CREATED)
async def create_item(item: Item):
    await item.save()
    return item

@router.get("/", response_model=List[Item])
async def get_items():
    items = await Item.objects.all()
    return items

@router.get("/{item_id}", response_model=Item)
async def get_item(item_id: UUID):
    item_db = await Item.objects.get(uuid=item_id)
    return item_db