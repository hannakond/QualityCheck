# This is a command line test for the API that performs quality check of the apples.
#
# (c) Hanna Kondrashova, University of London, 2023

# These modules will be used in the subsequent code to create a GUI, handle
# images, send API requests, and parse command-line arguments respectively.
import tkinter
from PIL import Image, ImageTk
import requests
import argparse

from src.log_utils import logger

"""
Colors associated with different classes of objects.
"""
CLASS_COLOR = {
    "good_apple": "green yellow",
    "rotten_apple": "orange red",
    "storage": "aquamarine"
}

def get_reply_from_api(file: str, url: str):
    """
    Utility function that takes a `file` and `url` and sends a request to the API that contains the file.
    """
    reply = requests.post(url, files={"file": open(file, 'rb')})
    return reply.json()

def draw_objects(reply, canvas):
    """
    Draws bounding boxes and class labels for object detections on a given canvas.

    Args:
        reply: A list of lists, where the first element is a list of detections.
        canvas: A tkinter canvas on which to draw the bounding boxes and class labels.

    Returns:
        None.
    """

    # YOLO always outputs an list of lists...
    if len(reply) != 1:
        logger.warn(f"No objects in reply {reply}")
        return

    # ... where the first element is a list of detections
    for detection in reply[0]:
        # Each detection is dictionary with `class_name`...
        class_name = detection["class_name"]
        # ... and bounding box
        bbox = detection["bbox"]
        logger.info(f"class_name {class_name}, bbox {bbox}")
        x1, y1, x2, y2 = bbox
        canvas.create_rectangle(
            x1, y1, x2, y2, width=2, outline=CLASS_COLOR[class_name])
        canvas.create_text((x1 + x2)/2, (y1 + y2)/2,
                            text=class_name, fill=CLASS_COLOR[class_name])

def run_viz(args):
    """
    Entry point of the test.
    Creates a visualization of an image and drawn objects on top of it.

    Args:
        args: An object containing command-line arguments.

    Returns:
        None.

    """
    tk = tkinter.Tk()
    img = Image.open(args.filename)
    img = ImageTk.PhotoImage(img)
    tk.geometry(f"{img.width()}x{img.height()}")
    canvas = tkinter.Canvas(tk, width=img.width(), height=img.height())
    canvas.pack()
    canvas.create_image(0, 0, image=img, anchor=tkinter.NW)

    draw_objects(get_reply_from_api(args.filename, args.url), canvas)

    tk.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="test_viz",
        description="QualityCheck test with visualization",
    )
    parser.add_argument("-f", "--filename", required=True,
                        help="Image file to process.")
    parser.add_argument("-u", "--url", help="QualityCheck processing URI.",
                        default="http://localhost:8000/detect")
    run_viz(parser.parse_args())