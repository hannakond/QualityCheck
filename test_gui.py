import tkinter
from PIL import Image, ImageTk
import requests

def get_coordinates(file: str):
    url = "http://localhost:8000/detect"
    reply = requests.post(url, files={"file": open(file, 'rb')})
    return reply.json()

if __name__ == "__main__":
    tk = tkinter.Tk()
    img = Image.open("example.jpg")
    img = ImageTk.PhotoImage(img)
    tk.geometry(f"{img.width()}x{img.height()}")
    canvas = tkinter.Canvas(tk, width=img.width(), height=img.height())
    canvas.pack()
    canvas.create_image(0, 0, image=img, anchor=tkinter.NW)

    reply = get_coordinates("example.jpg")
    if len(reply) == 1:
        for detection in reply[0]:
            class_name = detection["class_name"]
            bbox = detection["bbox"]
            print(class_name, bbox)
            x, y, x1, y1 = bbox
            canvas.create_rectangle(x, y, x1, y1, width=2, outline="red")
            canvas.create_text(x, y, text=class_name, fill="red")

    tk.mainloop()
