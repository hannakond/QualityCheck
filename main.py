import time
import uuid

from fastapi import (
    FastAPI, File, UploadFile, Request, Response
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from PIL import Image
from io import BytesIO

from src.log_utils import logger
from src.yolo_utils import payload_to_json, MODEL
from src.cuda_utils import get_cuda_details
from src.banner import BANNER

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Page shown by default when opening the root location.
    """
    return BANNER


@app.middleware("http")
async def log_requests(request: Request, call_next):
    uid = uuid.uuid4()
    logger.info(
        f"{uid}\tBEGIN\tmethod = {request.method}\trequest path = {request.url.path}")
    start = time.monotonic()

    response = await call_next(request)

    process_time = "{0:.5f}".format((time.monotonic() - start))
    logger.info(
        f"{uid}\tEND\tesponse time = {process_time}s\tstatus code = {response.status_code}")

    return response


@app.middleware("http")
async def catch_all_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        logger.exception(Exception)
        return Response("Internal server error", status_code=500)


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    results = MODEL(Image.open(BytesIO(await file.read())))

    return payload_to_json(results, MODEL)


@app.get("/status")
async def status():
    """
    Shows the host state details.
    """
    cuda_details = get_cuda_details()

    return {"status": "ok",
            "cuda": cuda_details}


if __name__ == "__main__":
    import uvicorn

    APP_STR = "main:app"
    uvicorn.run(APP_STR, host='0.0.0.0', port=8000, workers=1, reload=False)
