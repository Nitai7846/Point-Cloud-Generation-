#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 17:20:23 2023

@author: nitaishah
"""
import matplotlib.pyplot as plt
import torch
torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
repo = "isl-org/ZoeDepth"

model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

from PIL import Image
image = Image.open("/Users/nitaishah/Desktop/ZOEDEPTH/IMAGES/5.JPG").convert("RGB")  # load
depth_numpy = zoe.infer_pil(image)  # as numpy

depth_pil = zoe.infer_pil(image, output_type="pil")

depth_tensor = zoe.infer_pil(image, output_type="tensor")

from zoedepth.utils.misc import pil_to_batched_tensor
X = pil_to_batched_tensor(image).to(DEVICE)
depth_tensor = zoe.infer(X)



import PIL

from PIL import Image

image = Image.open("/Users/nitaishah/Desktop/Cracks/hairline-crack-wall (1).jpg")
depth = zoe.infer_pil(image)
plt.imshow(depth)
from zoedepth.utils.misc import save_raw_16bit

#fpath = "/Users/nitaishah/Downloads/ZOEDEPTH/output.png"
#save_raw_16bit(depth, fpath)

#from zoedepth.utils.misc import colorize

#colored = colorize(depth)
#fpath_colored = "/Users/nitaishah/Downloads/ZOEDEPTH/output/colored.png"
#Image.fromarray(colored).save(fpath_colored)

import numpy as np
import open3d as o3d


width, height = image.size
#depth_image = (depth * 255 / np.max(depth)).astype('uint8')

image = np.array(image)

depth_o3d = o3d.geometry.Image(depth)
image_o3d = o3d.geometry.Image(image)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)

camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(width, height, 2000, 2000, width/2, height/2)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
o3d.visualization.draw_geometries([pcd])



cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.0)
pcd = pcd.select_by_index(ind)

# estimate normals
pcd.estimate_normals()
pcd.orient_normals_to_align_with_direction()

# surface reconstruction
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, n_threads=1)[0]

# rotate the mesh
rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
mesh.rotate(rotation, center=(0, 0, 0))

# save the mesh
o3d.io.write_triangle_mesh(f'./mesh_4.obj', mesh)

# visualize the mesh
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

