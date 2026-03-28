import structlog
from fastapi import APIRouter

router = APIRouter()
logger = structlog.get_logger()


@router.get("/health")
async def health_check():
    logger.info("health_check")
    return {"status": "healthy", "service": "docmind"}
