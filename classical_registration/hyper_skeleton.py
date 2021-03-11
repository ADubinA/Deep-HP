import numpy as np
import copy
import open3d as o3d
import tqdm
import random
from sklearn.cluster import AgglomerativeClustering
import glob, os

class HyperSkeleton:
    def __init__(self, normal_rad =10):
        self.normal_rad = 10
        self.resample = 1
        self.atlas = o3d.geometry.LineSet()
    def load_image(self, path):
        pcd = o3d.io.read_point_cloud(path)

        numpy_source = np.asarray(pcd.points)
        numpy_source = numpy_source[numpy_source[:, 2] > 300]
        numpy_source = numpy_source[numpy_source[:, 2] < 501]

        pcd_index = np.random.randint(0, numpy_source.shape[0], int(numpy_source.shape[0] * self.resample))
        pcd.points = o3d.utility.Vector3dVector(numpy_source[pcd_index])
        pcd.paint_uniform_color([0.1, 0.1, 0.1])
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_rad, max_nn=30))

        return pcd

    def create_atlas(self):
        self.atlas.points = o3d.utility.Vector3dVector(np.array([[390, 320, 540], np.array([420, 260, 560])]))
        self.atlas.points = o3d.utility.Vector3dVector(np.array([[180, 270, 500], np.array([190, 200, 500])]))
        self.atlas.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
        self.atlas.lines = o3d.utility.Vector2iVector(np.array([[2, 3]]))

