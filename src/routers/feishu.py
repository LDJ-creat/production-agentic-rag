import logging
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse

from src.dependencies import FeishuDep

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feishu"])


@router.post("/feishu/events")
async def feishu_event_callback(
    payload: Dict[str, Any] = Body(...),
    feishu_service: FeishuDep,
):
    """Feishu event callback endpoint (URL verification + message events)."""
    if not feishu_service:
        raise HTTPException(status_code=503, detail="Feishu bot is not configured")

    result = await feishu_service.handle_event(payload)
    return JSONResponse(content=result)
