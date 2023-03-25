import torch
import os

def payload_to_json(results, model):
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

MODEL_PATH = os.path.join("model", "best.pt")
YOLO_PATH = "yolov5"
DEVICE = "cuda:0"
MODEL = torch.hub.load(os.path.join(os.getcwd(), YOLO_PATH), "custom",
                       path=MODEL_PATH, source="local", device=DEVICE, force_reload=True)
