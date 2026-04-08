from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import logging

from src.services.pnr_status import get_pnr_status

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PNR Service API",
    version="1.0.0",
    description="API to fetch PNR status and upgrade class"
)

# ------------------ Response Models ------------------ #

class PNRResponse(BaseModel):
    pnr: str
    status: str


# ------------------ Routes ------------------ #

@app.get("/", tags=["Health"])
def health_check():
    return {"message": "API is running"}


@app.get("/pnr", response_model=PNRResponse, tags=["PNR"])
def get_pnr(
    pnr: str = Query(..., min_length=10, max_length=10, description="10-digit PNR number")
):
    try:
        logger.info(f"Fetching PNR status for {pnr}")
        result = get_pnr_status(pnr)

        if not result:
            raise HTTPException(status_code=404, detail="PNR not found")

        return PNRResponse(pnr=pnr, status=result)

    except Exception as e:
        logger.error(f"Error fetching PNR: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/pnr/upgrade", response_model=PNRResponse, tags=["PNR"])
def upgrade_class(
    pnr: str = Query(..., min_length=10, max_length=10)
):
    try:
        logger.info(f"Checking upgrade possibility for {pnr}")
        result = get_pnr_status(pnr)

        if not result:
            raise HTTPException(status_code=404, detail="PNR not found")

        # Placeholder logic (replace with real upgrade logic)
        upgraded_status = f"Upgrade check: {result}"

        return PNRResponse(pnr=pnr, status=upgraded_status)

    except Exception as e:
        logger.error(f"Error upgrading class: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")