import os,sys
import numpy as np
import torch
# 将 'vggt/vggt' 添加到 sys.path 中
sys.path.append(os.path.join(os.path.dirname(__file__), 'vggt'))
from safetensors.torch import load_file
from vggt.models.vggt import VGGT, Infraeyes
from vggt.utils.geometry import closed_form_inverse_se3
from vggt.utils.load_fn import load_and_preprocess_images_camera, load_and_preprocess_images
from carla import read_params_from_json

os.makedirs("onnx", exist_ok=True)

device = "cuda"
model = Infraeyes().to(device)
# state_dict = load_file("/home/qity/Documents/phs/infra_eye/outputs/Infraeye-carla-finetune/final-checkpoint/model.safetensors")
# model.load_state_dict(state_dict, strict=True)
image_names = [os.path.join("examples", "carla", "town10_depth", "020918", "rgb", f"camera_{i}.png") for i in range(8)]
caminfo_path = "examples/carla/town10_depth/params"
caminfo_files = sorted(os.listdir(caminfo_path))

intrinsics, extrinsics = read_params_from_json(root_path=caminfo_path, files=caminfo_files, if_scale=False)
intrinsics = np.stack(intrinsics)
extrinsics = np.stack(extrinsics)
    

images, intrinsics, extrinsics = load_and_preprocess_images_camera(image_names, intrinsics, extrinsics, mode= "crop")
extrinsics = extrinsics[None][:,:, :3, :4]  # Convert to (B, N, 3, 4) format
old_extrinsics = extrinsics.clone()

B, S, _, _ = extrinsics.shape
extrinsics_homog = torch.cat(
    [
        extrinsics,
        torch.zeros((B, S, 1, 4), dtype=extrinsics.dtype, device=extrinsics.device),
    ],
    dim=-2,
)
extrinsics_homog[:, :, -1, -1] = 1.0

first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)
cam_centers = new_extrinsics[:, :, :3, 3]  # (B, S, 3)
ref_cam = cam_centers[:, 0:1, :]  # (B,1,3)
rel_distances = torch.norm(cam_centers - ref_cam, dim=-1)  # (B, S)
scale = rel_distances.mean(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1)
new_extrinsics[:, :, :3, 3] /= scale.unsqueeze(-1)
new_extrinsics = new_extrinsics[:, :, :3]
        
images = images.to(device)
intrinsics = intrinsics.to(device)
new_extrinsics = new_extrinsics.to(device)

input_names = ["input_images","extrinsics", "intrinsics"]
inputs = (images, new_extrinsics, intrinsics)
inputs = tuple(inp.to(dtype=torch.float32) for inp in (images, new_extrinsics, intrinsics))

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
        inputs,
        "onnx/infraeyes.onnx",
        input_names=input_names,
        output_names=output_names,
        # dynamic_axes={name: {0: "num_images"} for name in input_names + output_names},
        dynamo=True,
    )

    # with torch.amp.autocast(device, dtype=torch.float16):
    #     torch.onnx.export(
    #         model,
    #         inputs,
    #         "onnx_fp16/infraeyes_fp16.onnx",
    #         input_names=input_names,
    #         output_names=output_names,
    #         dynamic_axes={
    #             name: {0: "num_images"} for name in input_names + output_names
    #         },
    #         dynamo=True
    #     )
