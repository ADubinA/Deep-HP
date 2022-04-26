import json
from datetime import datetime

import open3d as o3d
import numpy as np
from classical_registration.feature_hyper_skeleton import ABSFeatures, AffinePcdTransform
import os.path as osp
import os, glob
from classical_registration.utils import NumpyEncoder,chamfer_distance

class FPFH(ABSFeatures):
    def __init__(self, path, bounds=(np.array([0,0,0]), np.array([1,1,1])),
                 voxel_down_sample_size=8, has_labels=True, has_centers=True):
        super().__init__(path, bounds, has_labels, has_centers, voxel_down_sample_size)
        self.cluster_func = None
    def create_global_features(self):
        radius_normal = self.voxel_down_sample_size * 2
        self.down_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = self.voxel_down_sample_size * 5
        self.feature_pcd = o3d.pipelines.registration.compute_fpfh_feature(
            self.down_pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        self.features = self.down_pcd.points
    def register(self, target):
        distance_threshold = 10#self.voxel_down_sample_size * 0.5
        print(":: Apply fast global registration with distance threshold %.3f" \
              % distance_threshold)
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            self.down_pcd, target.down_pcd, self.feature_pcd, target.feature_pcd,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        if result.correspondence_set is None:
            fits = []
        else:
            fits  = [fit for fit in np.asarray(result.correspondence_set)]

        return AffinePcdTransform(result.transformation), result.inlier_rmse, fits



if __name__ == "__main__":

    time_string = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    target_path = r"D:\datasets\VerSe2020\new_train\sub-verse506.ply"
    folder_path = r"D:\datasets\VerSe2020\new_train\\"
    figuring_fpfh(target_path=target_path,
              source_folder_path=folder_path,
              save_path=fr"D:\research_results\HyperSkeleton\{time_string}")


