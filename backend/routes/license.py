"""
License activation routes.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config import VelquaConfig as Config
from backend.license import LicenseManager, LicenseStatus
from backend.logging_config import get_logger

logger = get_logger("routes.license")

router = APIRouter(prefix="/license", tags=["license"])

# Singleton license manager
_manager = LicenseManager(Config.DATA_DIR)


def get_license_manager() -> LicenseManager:
    """Access the shared license manager."""
    return _manager


class ActivateRequest(BaseModel):
    key: str


@router.post("/activate")
async def activate_license(req: ActivateRequest):
    """Validate a license key and store the activation."""
    result = await _manager.activate(req.key)
    return {
        "success": result.success,
        "status": result.status.value,
        "message": result.message,
        "customer_email": result.customer_email,
        "product_name": result.product_name,
    }


@router.get("/status")
async def license_status():
    """Check current license status."""
    result = _manager.check()
    return {
        "status": result.status.value,
        "message": result.message,
        "is_active": result.status in (LicenseStatus.ACTIVE, LicenseStatus.TRIAL),
        "is_trial": result.status == LicenseStatus.TRIAL,
        "customer_email": result.customer_email,
    }


@router.post("/deactivate")
async def deactivate_license():
    """Remove the license activation (returns to trial mode)."""
    success = _manager.deactivate()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to deactivate license")
    return {"success": True, "message": "License deactivated. Running in trial mode."}


@router.post("/revalidate")
async def revalidate_license():
    """Re-validate the cached license online."""
    result = await _manager.revalidate()
    return {
        "success": result.success,
        "status": result.status.value,
        "message": result.message,
    }
