# -*- coding: utf-8 -*-
# This is the entry point of the server application that 
# performs quality check of the apples.
#
# (c) Hanna Kondrashova, University of London, 2023


# time and uuid are used in log_requests middleware to calculate the time taken by the request. 
import time
import uuid

# FastAPI is the main class used to build the application. File,
# UploadFile, Request, and Response are different request and response objects
# provided by FastAPI. 
from fastapi import (
    FastAPI, File, UploadFile, Request, Response
)
# CORSMiddleware is the middleware used to handle cross-origin requests.
from fastapi.middleware.cors import CORSMiddleware

# HTMLResponse is the response object used to return HTML content. 
from fastapi.responses import HTMLResponse

# Image and BytesIO are used to read and process images.
from PIL import Image
from io import BytesIO

# Custom classes and functions from 'src' folder.
from src.log_utils import logger
from src.yolo_utils import payload_to_json, MODEL
from src.cuda_utils import get_cuda_details
from src.banner import BANNER

app = FastAPI()

# Add CORS middleware to allow cross-origin requests from any origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Displays a banner as the default page when opening the root location.

    Returns:
        The banner as an HTML response.
    """
    return BANNER


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Logs information about incoming HTTP requests and their responses.

    Args:
        request: The incoming HTTP request.
        call_next: The function to call to proceed with the request.

    Returns:
        The HTTP response.
    """

    # Generate a unique identifier for the request
    uid = uuid.uuid4()

    # Log the beginning of the request
    logger.info(
        f"{uid}\tBEGIN\tmethod = {request.method}\trequest path = {request.url.path}")

    # Record the start time of the request processing
    start = time.monotonic()

    # Proceed with the request and get the response
    response = await call_next(request)

    # Calculate the processing time of the request
    process_time = "{0:.5f}".format((time.monotonic() - start))

    # Log the end of the request, along with the processing time and response status code
    logger.info(
        f"{uid}\tEND\tesponse time = {process_time}s\tstatus code = {response.status_code}")

    # Return the response
    return response


@app.middleware("http")
async def catch_all_exceptions(request: Request, call_next):
    """
    A middleware that catches all exceptions thrown during request processing.

    Args:
        request: The incoming HTTP request.
        call_next: The function to call to proceed with the request.

    Returns:
        The HTTP response, or an error response if an exception was thrown.
    """

    try:
        # Proceed with the request and get the response
        return await call_next(request)
    except Exception:
        # If an exception is thrown, log it and return an error response
        logger.exception(Exception)
        return Response("Internal server error", status_code=500)


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Runs an object detection model on the provided image file.

    Args:
        file: The image file to run the model on.

    Returns:
        A JSON payload containing the results of the object detection model.
    """
    # Open the image file and run it through the model
    results = MODEL(Image.open(BytesIO(await file.read())))

    # Convert the results to JSON format and return them
    return payload_to_json(results, MODEL)


@app.get("/info")
async def info():
    """
    Returns information about the host system, such as CUDA details.
    """
    # Get the CUDA details of the host system
    cuda_details = get_cuda_details()

    # Return a JSON payload with the status and CUDA details
    return {"status": "ok",
            "cuda": cuda_details}


if __name__ == "__main__":
    import uvicorn

    APP_STR = "quality_check:app"
    uvicorn.run(APP_STR, host='0.0.0.0', port=8000, workers=1, reload=False)
