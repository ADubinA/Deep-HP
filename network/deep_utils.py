from random import random

import open3d as o3d
import torch
import numpy as np
from matplotlib.colors import hsv_to_rgb
import os.path as osp
import pathlib

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

def gen_colormap(num_classes):
    colormap = np.zeros((num_classes, 3))
    colormap[..., 0] = np.arange(num_classes)/num_classes
    colormap[..., 1] = 1
    colormap[..., 2] = 1
    colormap = hsv_to_rgb(colormap)
    return colormap


def visualize_graph(g, has_colors=True, return_value=False):
    linespace = o3d.geometry.LineSet()

    point_indexed = {node: i  for i, node in enumerate(g.nodes)}
    linespace.points = o3d.utility.Vector3dVector([g.nodes[i]["point"] for i in g.nodes])
    linespace.lines = o3d.utility.Vector2iVector([(point_indexed[edge[0]],point_indexed[edge[1]]) for edge in g.edges])

    if has_colors:
        linespace.colors = o3d.utility.Vector3dVector(
            [(g.nodes[point_indexed[edge[0]]].get("color",np.array([0,0,0]))
              + g.nodes[point_indexed[edge[1]]].get("color",np.array([0,0,0])))/2
              for edge in g.edges])
    if return_value:
        return linespace
    else:
        o3d.visualization.draw([linespace])

def visualize_batch_result(data:Data, results=None, num_classes=64, save_path=None):
    if save_path:
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    data_list = data.to_data_list()
    for i, pcd_data in enumerate(data_list):
        if results is not None:
            labels = results[data.batch == i].cpu().numpy()
        else:
            labels=None
        if save_path:
            file_path = osp.join(save_path, f"{i}.ply")
        else:
            file_path = None

        visualize_single_result(pcd_data, labels, num_classes=num_classes,save_path=file_path)


def visualize_single_result(data, labels=None, num_classes=64, save_path=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(data.pos.cpu()))

    if labels is not None:
        color_map = gen_colormap(num_classes)
        colors = np.asarray(color_map[labels, :])
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))

    if save_path:
        dir = osp.join(save_path)
        o3d.io.write_point_cloud(dir, pcd)
    else:
        o3d.visualization.draw([pcd])

def extra_pretty_vis(list_geo, rad = 3):
    vis = []
    for geo in list_geo:
        points = np.asarray(geo.points)
        mesh = o3d.geometry.TriangleMesh()
        for i, point in enumerate(points):
            sphere = o3d.geometry.TriangleMesh().create_sphere(radius=rad)
            sphere.translate(point)
            if geo.has_colors():
                sphere.paint_uniform_color(np.asarray(geo.colors)[i])
            mesh += sphere
        mesh.compute_vertex_normals()
        vis.append(mesh)
        try:
            if geo.has_lines():
                vis.append(geo)
        except AttributeError:
            continue
    o3d.visualization.draw(vis)



class RandomCrop(BaseTransform):
    "assumes normalized pointclouds. will crop a random sub cloud"
    def __init__(self, block_size_min, block_size_max):
        super().__init__()
        self.block_size_min = block_size_min
        self.block_size_max = block_size_max

    def __call__(self, data: Data) -> Data:
        finished = False
        while not finished:
            block_size = (self.block_size_max - self.block_size_min) * torch.rand((3)) + self.block_size_min
            pos = data["pos"]
            maxi = torch.tensor((1, 1, 1))
            mini = torch.tensor((-1, -1, -1))
            start = torch.max((maxi - block_size - mini) * torch.rand((3)) + mini, mini)
            end = torch.min(start + block_size, maxi)
            filtered = torch.logical_and(torch.all(data.pos > start, axis=1), torch.all(data.pos < end, axis=1))
            new_data = data.clone()
            new_data.pos = data.pos[filtered]
            new_data.x = data.x[filtered]
            new_data.y = data.y[filtered]

            if new_data.num_nodes>1000:
                finished = True
        return new_data

class RandomScaleAxis(BaseTransform):
    def __init__(self, scales, axis):
        self.scales = scales
        self.axis= axis
    def __call__(self, data: Data) -> Data:
        scale = random.uniform(*self.scales)
        data.pos[:,self.axis] = data.pos[:,self.axis] * scale
        return data

