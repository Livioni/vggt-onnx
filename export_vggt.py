import os

import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cpu"
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
image_names = [os.path.join("examples", "kitchen", "images", f"{i:02}.png") for i in [0, 1]]
images = load_and_preprocess_images(image_names, "pad").to(device)

input_names = ["input_images"]
output_names = [
    "pose_enc",
    "depth",
    "depth_conf",
    "world_points",
    "world_points_conf",
    "images",
]
with torch.no_grad():
    torch.onnx.export(
        model,
        images,
        "vggt-onnx/vggt.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={name: {0: "num_images"} for name in input_names + output_names},
    )
