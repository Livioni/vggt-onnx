import os

import numpy as np
import onnxruntime as ort
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def assert_similar(a, b, delta=1e-3):
    assert a.shape == b.shape
    diff_99p = np.percentile(np.abs(a - b), 99.0)
    print(f"    {diff_99p}")
    assert diff_99p < delta


def compare_torch_onnx(torch_pred, onnx_pred, ort_sess, delta, conf_delta):
    for output_name in torch_pred:
        print(f"    Checking for similar {output_name}")
        idx = [x.name for x in ort_sess._outputs_meta].index(output_name)
        assert_similar(
            predictions[output_name],
            onnx_pred[idx],
            conf_delta if output_name.endswith("_conf") else delta,
        )


MAX_NUM_IMAGES = 3

image_names = [
    os.path.join("vggt", "examples", "kitchen", "images", f"{i:02}.png")
    for i in range(MAX_NUM_IMAGES)
]
images = load_and_preprocess_images(image_names, "pad")
model = VGGT.from_pretrained("facebook/VGGT-1B")
ort_sess = ort.InferenceSession("vggt.onnx")
ort_sess_fp16 = ort.InferenceSession("vggt_fp16.onnx")

for num_images in range(1, MAX_NUM_IMAGES + 1):
    print(f"Checking {num_images} input images")
    print("  Running PyTorch model")
    input_images = images[:num_images]
    with torch.no_grad():
        predictions = model(input_images)

    print("  Running fp32 ONNX model")
    outputs = ort_sess.run(None, {"input_images": input_images.numpy()})
    compare_torch_onnx(predictions, outputs, ort_sess, 1e-4, 1e-3)

    print("  Running fp16 ONNX model")
    outputs = ort_sess_fp16.run(None, {"input_images": input_images.numpy()})
    compare_torch_onnx(predictions, outputs, ort_sess, 1e-2, 1e-1)
