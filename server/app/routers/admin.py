from fastapi import APIRouter, Depends
from app.utils.auth import get_tenant_id
import psutil
import time

router = APIRouter()

# Store server start time
START_TIME = time.time()


@router.get("/health", tags=["Administration"])
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {"status": "ok", "uptime": time.time() - START_TIME}


@router.get("/metrics", tags=["Administration"])
async def metrics(_: str = Depends(get_tenant_id)):
    """
    Metrics endpoint for monitoring.
    Requires authentication as it exposes system information.
    """
    # Get basic system metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    return {
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_mb": memory.used / (1024 * 1024),
            "memory_total_mb": memory.total / (1024 * 1024),
            "disk_percent": disk.percent,
            "disk_used_gb": disk.used / (1024 * 1024 * 1024),
            "disk_total_gb": disk.total / (1024 * 1024 * 1024),
        },
        "uptime": time.time() - START_TIME,
    }
