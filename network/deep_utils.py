# from random import random
import copy

import open3d as o3d
import torch
import numpy as np
from matplotlib.colors import hsv_to_rgb
import os.path as osp
import pathlib
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from emd_build.emd_module import emdModule
from math import floor

def emd_cuda(pcd1, pcd2):
    # to cuda
    pcd1, pcd2 = torch.tensor(pcd1).cuda(), torch.tensor(pcd2).cuda()

    # resample to both to 1024*n
    sample_times = min(floor(pcd1.shape[0] / 1024), floor(pcd2.shape[0] / 1024))
    sample_num = 4*1024#sample_times*1024
    pcd1 = pcd1[torch.randperm(pcd1.shape[0])[:sample_num]]
    pcd2 = pcd2[torch.randperm(pcd2.shape[0])[:sample_num]]

    # run emd
    pcd1, pcd2 = pcd1.unsqueeze(0), pcd2.unsqueeze(0)
    emd = emdModule()
    dis, ass = emd(pcd1, pcd2, 0.05, 3000)
    return np.sqrt(dis.cpu())

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

def extra_pretty_vis(list_geo, rad = 5, lookat = [200, 200, 100],eye=[-200,50,100], up=[0,0,1]):
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
    o3d.visualization.draw(vis,lookat = lookat,eye=eye, up=up, show_skybox=False)

def visualize_registration_results(source_path, target_path, moved_path):
    scale_array = np.array([[1,0,0,0],[0,1,0,0],[0,0,4,0],[0,0,0,1]])
    target = o3d.io.read_point_cloud(target_path)
    target.transform(scale_array)
    target.paint_uniform_color([1,0,0])
    source = o3d.io.read_point_cloud(source_path)
    source.transform(scale_array)
    source.paint_uniform_color([0,1,0])
    moved = o3d.io.read_point_cloud(moved_path)
    moved.paint_uniform_color([0,1,0])
    moved.transform(scale_array)
    source = source.voxel_down_sample(4)
    if "[[00_00_00]_[10_10_05]]" in moved_path:
        bounds = np.array([[0, 0, 0], [1, 1, 0.5]])
    else:
        bounds = np.array([[0, 0, 0.5], [1, 1, 1]])

    numpy_pcd = np.asarray(source.points)
    min_bounds = source.get_min_bound() + bounds[0] * (source.get_max_bound() - source.get_min_bound())
    max_bounds = source.get_min_bound() + bounds[1] * (source.get_max_bound() - source.get_min_bound())
    idx = np.logical_and(np.all(numpy_pcd > min_bounds, axis=1),
                         np.all(numpy_pcd < max_bounds, axis=1))
    if source.has_colors():
        source.colors = o3d.utility.Vector3dVector(np.asarray(source.colors)[idx])
        source.points = o3d.utility.Vector3dVector(numpy_pcd[idx])
    source.translate(-source.get_min_bound())
    source.translate((target.get_max_bound() - target.get_min_bound())/2)
    source.translate(np.array([-100,200,-50]))

    extra_pretty_vis([target,source,moved], lookat = [500, 500, 400],eye=[-700,500,400])




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
        scale = np.random.uniform(*self.scales)
        data.pos[:,self.axis] = data.pos[:,self.axis] * scale
        return data

if __name__ == "__main__":
    source_path = r"D:\datasets\VerSe2020\new_validation\sub-verse508.ply"
    target_path = r"D:\research_results\HyperSkeleton\new_article_results\04-27-2022_12-39-32\sub-verse508_ref.ply"
    moved_path = r"D:\research_results\HyperSkeleton\new_article_results\04-27-2022_12-39-32\sub-verse508_[[00_00_05]_[10_10_10]].ply"
    visualize_registration_results(source_path, target_path, moved_path)