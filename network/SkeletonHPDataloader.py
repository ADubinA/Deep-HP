import glob
import os.path as osp

from sklearn.cluster import KMeans
from torch.utils.data import Dataset
import open3d as o3d
import numpy as np
from torch_geometric.data import Dataset as GeoDataset, Data
import torch
from torch_geometric.io import read_ply


class HPSkeletonDataset(Dataset):
    def __init__(self, base_folder, npoints=2500, cluster_func=None, num_classes=64,box_size = (500,500,200)):
        self.base_folder = base_folder
        self.npoint = npoints
        self.pcd_files_paths = []
        self.raw_labels_paths = []
        self.num_classes = num_classes
        self.load_paths()
        self.cluster_func = cluster_func
        self.set_clustering_function()
        self.box_size = box_size
    def load_paths(self):
        self.pcd_files_paths = list(glob.glob(osp.join(self.base_folder, "*.ply")))
        self.raw_labels_paths = list(glob.glob(osp.join(self.base_folder, "*.npy")))

    def set_clustering_function(self):
        ref_index = 0
        bins_list = np.load(self.raw_labels_paths[ref_index])
        if self.cluster_func is None:
            flat_bins = bins_list.reshape(bins_list.shape[0], -1)
            self.cluster_func = KMeans(init="k-means++", n_clusters=self.num_classes, n_init=5)
            self.cluster_func.fit(flat_bins)

    def reselect_points(self, pcd):
        """doesn't do anything to the pcd, just get the wanted idxes"""
        # get a smaller box
        for i in range(100):
            min_bound = pcd.get_min_bound()
            max_bound = pcd.get_max_bound()
            if np.all(max_bound-self.box_size > min_bound):
                sample_start = np.array([np.random.randint(min_bound[i], max_bound[i]-self.box_size[i])for i in range(3)])
            else:
                sample_start = np.zeros(3)
            point_mask = np.logical_and(np.all(np.asarray(pcd.points) > sample_start, axis=1),
                                        np.all(np.asarray(pcd.points) < sample_start+self.box_size, axis=1))
            bounded_points = np.where(point_mask == 1)[0]
            try:
                return np.random.choice(bounded_points, self.npoint, replace=False)
            except ValueError:
                continue

    def augment_points(self, pcd_points):
        return pcd_points

    def __len__(self):
        return len(self.pcd_files_paths)
    def __getitem__(self, index):
        pcd = o3d.io.read_point_cloud(self.pcd_files_paths[index])
        pcd_points = np.asarray(pcd.points)
        raw_labels = np.load(self.raw_labels_paths[index])

        sample_idx = self.reselect_points(pcd)
        pcd_points = pcd_points[sample_idx, :]
        raw_labels = raw_labels[sample_idx, :, :]

        labels = self.cluster_func.predict(raw_labels.reshape(-1, 64))
        return {"pcd": pcd_points, "labels": labels, "file_name": self.pcd_files_paths[index]}


class GeometricHPSDataset(GeoDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, cluster_func=None, num_classes=64):
        self.num_classes = num_classes
        self.cluster_func = cluster_func
        super().__init__(root, transform, pre_transform, pre_filter)
    @property
    def raw_file_names(self):
        paths = list(glob.glob(osp.join(self.raw_dir, "*.ply")))
        paths_names = [osp.splitext(osp.basename(path))[0] for path in paths]
        return paths_names

    @property
    def processed_file_names(self):
        names = self.raw_file_names
        names = [name+".pt" for name in names]
        return names

    def set_clustering_function(self):
        ref_index = 0
        bins_list = np.load(self.raw_paths[ref_index]+"_hist.npy")
        if self.cluster_func is None:
            flat_bins = bins_list.reshape(bins_list.shape[0], -1)
            self.cluster_func = KMeans(init="k-means++", n_clusters=self.num_classes, n_init=5)
            self.cluster_func.fit(flat_bins)


    def process_single(self, raw_path):
        pcd = o3d.io.read_point_cloud(raw_path + ".ply")
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
        labels = np.load(raw_path + "_hist.npy")
        y = self.cluster_func.predict(labels.reshape(-1, 64))
        data = Data(x=torch.tensor(np.asarray(pcd.normals)).float(),
                    pos=torch.tensor(np.asarray(pcd.points)).float(),
                    y=torch.tensor(y).float())  # ,num_nodes=len(pcd.points))

        if self.pre_filter is not None and not self.pre_filter(data):
            return None

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        return data
    def process(self):
        self.set_clustering_function()
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            name = osp.basename(raw_path)
            data = self.process_single(raw_path)
            if not data:
                continue

            torch.save(data, osp.join(self.processed_dir, f'{name}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_paths[idx])
        return data