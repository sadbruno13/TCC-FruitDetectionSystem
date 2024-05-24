from ultralytics import YOLO
import numpy as np
import base64
from PIL import Image
from io import BytesIO

class YOLOModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        self.model = YOLO(self.model_path)

    def detect_objects(self, imageb64):

        image_data = base64.b64decode(imageb64)
        image = Image.open(BytesIO(image_data))

        results = self.model(image)  # predict on an image

        names_dict = results[0].names
        probs = results[0].probs.data.numpy()
        objects = names_dict[np.argmax(probs)] 
        print(objects)
        return str(objects)