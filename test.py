import onnx

model = onnx.load("onnx/infraeyes.onnx")
for node in model.graph.node:
    for attr in node.attribute:
        if attr.HasField("t"):
            tensor = attr.t
            if tensor.data_type == onnx.TensorProto.DOUBLE:
                print(f"Found double tensor in node {node.name}")