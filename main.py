from fastapi import FastAPI
from src.services.pnr_status import get_pnr_status

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.get("/get_pnr")
def get_pnr(pnr: str):
    result = get_pnr_status(pnr)
    return {
        "pnr": pnr,
        "status": result
    }


@app.get("/upgrade_class")
def upgrade_class(pnr: str) :
    result = get_pnr_status(pnr)
    return {
        "pnr": pnr,
        "status": result
    }

