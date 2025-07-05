from shutil import rmtree

import onnx

model = onnx.load("vggt-onnx/vggt.onnx", load_external_data=True)
rmtree("vggt-onnx")
onnx.save_model(
    model,
    "vggt.onnx",
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="vggt.onnx_data",
    size_threshold=0,
)
onnx.save(model, "vggt.onnx")
