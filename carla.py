import sys
sys.path.append('.')
import os
import numpy as np
import json
from copy import copy

from scipy.spatial.transform import Rotation as R
import cv2

def scale_intrinsics(K, old_size, new_size):
    new_width, new_height = new_size
    old_width, old_height = old_size
    scale_x = new_width / old_width
    scale_y = new_height / old_height

    K_scaled = copy(K) 
    K_scaled[0,0] *= scale_x
    K_scaled[0,2] *= scale_x
    K_scaled[1,1] *= scale_y
    K_scaled[1,2] *= scale_y
    
    return K_scaled

def pose_unreal2opencv(c2w_mat):
    translation = c2w_mat[:3, 3]
    rot = R.from_matrix(c2w_mat[:3, :3])
    rot_vec = rot.as_rotvec()

    rot_vec_new = rot_vec[[1, 2, 0]]
    rot_vec_new[0] *= -1
    rot_vec_new[2] *= -1

    rot = R.from_rotvec(rot_vec_new)
    
    translation_new = translation[[1, 2, 0]]
    translation_new[1] *= -1

    c2w_mat = np.eye(4)
    c2w_mat[:3, :3] = rot.as_matrix()
    c2w_mat[:3, 3] = translation_new

    rot = np.eye(4)
    rot[1,1]=-1
    rot[2, 2] = -1
    c2w_mat =  rot @ c2w_mat
    return c2w_mat

# 需要确保 PNG_SCALE 与保存时一致，假设 PNG_SCALE = 1000.0
MAX_DEPTH_METERS = 1000.0          # CARLA 默认深度上限
PNG_SCALE       = 65535.0 / MAX_DEPTH_METERS 

def read_depth_png(file_path: str) -> np.ndarray:
    """
    从 PNG 图片中读取深度数据并转换为深度矩阵（单位：米）
    
    :param file_path: PNG 图片文件路径
    :return: 深度矩阵，单位米
    """
    # 读取PNG图像，注意读取为灰度图（0 - 65535）
    depth_uint16 = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    
    # 确保读取的图像是uint16类型
    if depth_uint16 is None or depth_uint16.dtype != np.uint16:
        raise ValueError("无法读取深度PNG图像，确保图像为PNG格式且包含uint16深度数据")
    
    # 将 uint16 转换为深度值（单位：米），除以 PNG_SCALE 来还原
    depth_meters = depth_uint16.astype(np.float32) / PNG_SCALE
    
    return depth_meters

def read_params_from_json(root_path, files, if_scale=False, old_size=(1920,1080), new_size=(512, 288)):
    intrinsics = []
    extrinsics = []
    for parmas_file in files:
        file_path = os.path.join(root_path, parmas_file)
        # 读取 JSON
        with open(file_path, "r") as f:
            data = json.load(f)
        K = np.around(np.array(data["intrinsic"]["K"]),decimals=4)
        T = np.around(np.array(data["extrinsic"]["T"]),decimals=4)
        if if_scale:
            K = scale_intrinsics(K, old_size, new_size)
        T = pose_unreal2opencv(T)
        intrinsics.append(K)
        extrinsics.append(T)
    return intrinsics, extrinsics
