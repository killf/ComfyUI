from PIL import Image
import numpy as np
import hashlib
import base64
import torch
import time
import cv2
import os

import folder_paths

class InputNode:
    FUNCTION = "handler"
    CATEGORY = "input"

    def handler(self, value):
        return value,

class InputString(InputNode):
    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"value": ("STRING", {"default": ""})}}

class InputText(InputNode):
    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"value": ("STRING", {"multiline": True})}}

class InputInt(InputNode):
    RETURN_TYPES = ("INT",)

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"value": ("INT", {"default": 0})}}

class InputFloat(InputNode):
    RETURN_TYPES = ("FLOAT",)

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"value": ("FLOAT", {"default": 0.0})}}

class InputImage(InputNode):
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required": {"image": (sorted(files), {"image_upload": True})}}

    def handler(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return image,
    
    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

class OutputNode:
    RETURN_TYPES = ()
    FUNCTION = "handler"
    OUTPUT_NODE = True
    CATEGORY = "output"

    @classmethod
    def IS_CHANGED(s, images, **kwargs):
        return time.time_ns()

class OutputImage(OutputNode):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE", )}}

    def handler(self, images):
        results = list()
        for img in images:
            img = 255. * img.cpu().numpy()
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            buf = cv2.imencode(".jpg", img)[1].tobytes()
            buf = base64.b64encode(buf).decode("utf8")
            results.append({
                "base64": buf,
                "prefix": "data:image/jpeg;base64,"
            })
        return { "ui": { "images": results } }

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "InputInt": InputInt,
    "InputFloat": InputFloat,
    "InputString": InputString,
    "InputText": InputText,
    "InputImage": InputImage,

    "OutputImage": OutputImage
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "InputInt": "Input Int",
    "InputFloat": "Input Float",
    "InputString": "Input String",
    "InputText": "Input Text",
    "InputImage": "Input Image",

    "OutputImage": "Output Image"
}
