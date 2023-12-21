from PIL import Image, ImageOps
import numpy as np
import requests
import hashlib
import base64
import torch
import time
import cv2
import os
import io

import folder_paths

class InputNode:
    FUNCTION = "handler"
    CATEGORY = "input"

    COLOR = "#432"
    BGCOLOR = "#653"

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
    RETURN_TYPES = ("IMAGE", "MASK")

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required": {"image": (sorted(files), {"image_upload": True})}}

    def handler(self, image):
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                image = requests.get(image).content
                image = Image.open(io.BytesIO(image))
            else:                
                image_path = image if os.path.isabs(image) else folder_paths.get_annotated_filepath(image)
                image = Image.open(image_path)
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        else:
            raise Example("The source of image doest not support.")
        
        i = ImageOps.exif_transpose(image)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask.unsqueeze(0))
    
    @classmethod
    def IS_CHANGED(s, image):
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                return image
            else:
                image_path = image if os.path.isabs(image) else folder_paths.get_annotated_filepath(image)
                image = open(image_path, 'rb').read()
        elif isinstance(image, bytes):
            pass
        else:
            return False

        m = hashlib.sha256()
        m.update(image)
        return m.digest().hex()
    
    @classmethod
    def VALIDATE_INPUTS(s, image):
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                return True
            else:
                if not os.path.isabs(image):
                    if not folder_paths.exists_annotated_filepath(image):
                        return "Invalid image file: {}".format(image)
        return True

class InputLoRA(InputNode):
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("Base Model", "LoRA Model")

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"value": ("STRING", {"default": ""})}}

    def handler(self, value):
        if "@" in value:
            loRAModel, baseModel = value.split("@")
        else:
            server = os.environ.get("ONESHOTPIPE_URL", "https://int-enterlive-oneshotpipe.xiaoice.com").rstrip("/")
            resp = requests.get(f"{server}/Image2Image/lora?LoRAId={value}")
            if resp.status_code != 200:
                raise Exception(f"The LoRA is invalid: {value}")
            
            resp = resp.json()
            baseModel, loRAModel = resp['BaseModel'], value
        
        baseModel = baseModel if os.path.isabs(baseModel) else f"/mnt/cpfs_sd/base_models/{baseModel}.safetensors"
        loRAModel = loRAModel if os.path.isabs(loRAModel) else f"/mnt/cpfs_sd/lora/{loRAModel}.safetensors"
        return baseModel, loRAModel

class OutputNode:
    RETURN_TYPES = ()
    FUNCTION = "handler"
    OUTPUT_NODE = True
    CATEGORY = "output"

    COLOR = "#223"
    BGCOLOR = "#335"

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
    "InputLoRA": InputLoRA,

    "OutputImage": OutputImage
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "InputInt": "Input Int",
    "InputFloat": "Input Float",
    "InputString": "Input String",
    "InputText": "Input Text",
    "InputImage": "Input Image",
    "InputLoRA": "Input LoRA",

    "OutputImage": "Output Image"
}
