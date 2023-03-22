import time
import random
import string
import os
from log_utils import logger

from fastapi import FastAPI, File, UploadFile, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from PIL import Image
from io import BytesIO
import torch

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
    return """
<!DOCTYPE html>
<html>
 <head>
  <meta charset="utf-8">
  <title>Quality Check API</title>
  <style>
  h1 { color: navy; text-align: center; }
  </style>
 </head>
 <body>
  <h1>API For Quality Check</h1>
  <ol>
    <li><a href="/docs">API swagger documentation</a></li>
    <li><a href="/status">Status of the host</a></li>
  </oi>
 </body>
</html>
"""

@app.middleware("http")
async def log_requests(request: Request, call_next):
    id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    logger.info(
        f"id={id} start, method={request.method} request path={request.url.path}")
    start_time = time.monotonic()

    response = await call_next(request)

    process_time = '{0:.2f}'.format((time.time() - start_time) * 1000)
    logger.info(
        f"id={id} end, execution_time={process_time}ms status_code={response.status_code}")

    return response


@app.middleware('http')
async def catch_all_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        logger.exception(Exception)
        return Response("Internal server error", status_code=500)


MODEL_PATH = os.path.join('model', 'best.pt')
YOLO_PATH = 'yolov5'
DEVICE = 'cuda:0'  # TODO check that device exist?
model = torch.hub.load(os.path.join(os.getcwd(), YOLO_PATH), 'custom',
                       path=MODEL_PATH, source='local', device=DEVICE, force_reload=True)


@app.options("/detect")
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    results = model(Image.open(BytesIO(await file.read())))

    return yolo_payload_to_json(results, model)


@app.get("/status")
async def api_status():
    """
    Shows the most important detials related to the host
    """
    cuda_avaliable = torch.cuda.is_available()
    if cuda_avaliable:
        cuda_detail = {
            "torch.cuda.is_available()": cuda_avaliable,
            "torch.cuda.device_count()": torch.cuda.device_count(),
            "torch.cuda.device(current_device)": torch.cuda.device(torch.cuda.current_device()),
            "torch.cuda.get_device_name(current_device)": torch.cuda.get_device_name(0)}
    else:
        cuda_detail = {"torch.cuda.is_available()": cuda_avaliable}

    return {"status": "ok",
            "cuda": cuda_detail}


def yolo_payload_to_json(results, model):
    """
    Encode YOLO output as JSON to output from the API.
    """
    return [
        [
            {
                "class": int(pred[5]),
                "class_name": model.model.names[int(pred[5])],
                "bbox": [int(x) for x in pred[:4].tolist()],
                "confidence": float(pred[4]),
            }
            for pred in result
        ]
        for result in results.xyxy
    ]


if __name__ == "__main__":
    import uvicorn

    APP_STR = "main:app"
    uvicorn.run(APP_STR, host='0.0.0.0', port=8000, workers=1, reload=False)
