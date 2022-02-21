import open3d as o3d
import torch
import numpy as np
from matplotlib.colors import hsv_to_rgb
import os.path as osp
import pathlib


def gen_colormap(num_classes):
    colormap = np.zeros((num_classes, 3))
    colormap[..., 0] = np.arange(num_classes)/num_classes
    colormap[..., 1] = 1
    colormap[..., 2] = 1
    colormap = hsv_to_rgb(colormap)
    return colormap


def visualize_single_result(data,results, num_classes=64, save_path=None):
    if save_path:
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    data_list = data.to_data_list()
    for i, pcd_data in enumerate(data_list):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd_data.pos.cpu()))

        color_map = gen_colormap(num_classes)
        labels = results[data.batch == i].cpu().numpy()
        colors = np.asarray(color_map[labels,:])
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))

        if save_path:
            dir = osp.join(save_path,f"{i}.ply")
            o3d.io.write_point_cloud(dir, pcd)
        else:
            o3d.visualization.draw([pcd])

