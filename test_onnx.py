import os

import numpy as np
import onnxruntime as ort
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def assert_similar(a, b, delta=1e-3):
    assert a.shape == b.shape
    assert np.abs(a - b).max() < delta


MAX_NUM_IMAGES = 3

image_names = [
    os.path.join("vggt", "examples", "kitchen", "images", f"{i:02}.png")
    for i in range(MAX_NUM_IMAGES)
]
images = load_and_preprocess_images(image_names, "pad")
model = VGGT.from_pretrained("facebook/VGGT-1B")
ort_sess = ort.InferenceSession("vggt.onnx")

for num_images in range(1, MAX_NUM_IMAGES + 1):
    print(f"Checking {num_images} input images")
    print("  Running PyTorch model")
    input_images = images[:num_images]
    with torch.no_grad():
        predictions = model(input_images)

    print("  Running ONNX model")
    outputs = ort_sess.run(None, {"input_images": input_images.numpy()})

    for output_name in predictions:
        print(f"  Checking for similar {output_name}")
        idx = [x.name for x in ort_sess._outputs_meta].index(output_name)
        assert_similar(predictions[output_name], outputs[idx])
