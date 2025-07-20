import os,sys

import numpy as np
import onnxruntime as ort
import torch
import time  # 新增
sys.path.append(os.path.join(os.path.dirname(__file__), 'vggt'))
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
            torch_pred[output_name],
            onnx_pred[idx],
            conf_delta if output_name.endswith("_conf") else delta,
        )


MAX_NUM_IMAGES = 3

image_names = [
    os.path.join("vggt", "examples", "kitchen", "images", f"{i:02}.png")
    for i in range(MAX_NUM_IMAGES)
]
images = load_and_preprocess_images(image_names, "pad").to("cuda")

# for onnx_model, tol, conf_tol in [("onnx/infraeyes.onnx", 1e-4, 1e-3), ("onnx/infraeyes_fp16.onnx", 1e-2, 2e-1)]:、
onnx_model = "vggt.onnx"
print(f"Loading ONNX model {onnx_model}")
ort_sess = ort.InferenceSession(onnx_model, providers=['CUDAExecutionProvider'])
for num_images in range(1, MAX_NUM_IMAGES + 1):
    input_images = images[:num_images]
    print(f"  Running on {num_images} input images")
    input_numpy = input_images.cpu().numpy()
    
    # === 计时开始 ===
    start_time = time.time()
    output = ort_sess.run(None, {"input_images": input_numpy})
    end_time = time.time()
    # === 计时结束 ===
    
    elapsed = end_time - start_time
    print(f"    Inference time: {elapsed * 1000:.2f} ms")  # 毫秒输出

    # compare_torch_onnx(expected[num_images - 1], output, ort_sess, tol, conf_tol)
del ort_sess
