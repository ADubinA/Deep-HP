import json
from datetime import datetime

import open3d as o3d
import numpy as np
import copy
from utils import chamfer_distance
from feature_hyper_skeleton import ABSFeatures, ABSPcdTransform
import os.path as osp
import os, glob
from classical_registration.utils import NumpyEncoder

class FPFH(ABSFeatures):
    def __init__(self, path, bounds=(np.array([0,0,0]), np.array([1000,1000,1000])),
                 voxel_down_sample_size=8, has_labels=True, has_centers=True):
        super().__init__(path, bounds, has_labels, has_centers, voxel_down_sample_size)
    def create_global_features(self):
        radius_normal = self.voxel_down_sample_size * 2
        self.down_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = self.voxel_down_sample_size * 5
        self.feature_pcd = o3d.pipelines.registration.compute_fpfh_feature(
            self.down_pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    def register(self, target):
        distance_threshold = 10#self.voxel_down_sample_size * 0.5
        print(":: Apply fast global registration with distance threshold %.3f" \
              % distance_threshold)
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            self.down_pcd, target.down_pcd, self.feature_pcd, target.feature_pcd,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        print(result.transformation)
        quit()
        return result

def figuring_fpfh(target_path, source_folder_path, save_path,test_type="self", has_labels=True, has_centers=True):
    os.mkdir(os.path.join(save_path))
    target = FPFH(target_path, has_labels=has_labels, has_centers=has_centers)
    target.create_global_features()
    o3d.io.write_point_cloud(osp.join(save_path, osp.basename(target_path).replace(".ply", "_ref.ply")),
                             target.down_pcd)
    subvolume_size = 100
    losses = []
    for file_path in glob.glob(osp.join(source_folder_path, "*.ply")):
        if "center" in file_path or "label" in file_path or "gl" in file_path:
            continue
        print(file_path)
        for slicer_index in [0, 100, 200, 300, 400, 500, 600]:
            bounds = (np.array([-1000,-1000,slicer_index]), np.array([1000,1000,slicer_index+subvolume_size]))
            source = FPFH(file_path, bounds, has_labels=has_labels, has_centers=has_centers)
            if len(source.base_pcd.points) < 100:
                continue

            source.create_global_features()
            transform, loss, fit = source.register(target)
            if source.has_labels:
                segment_losses = source.calculate_label_metric(transform, target, "segment")
            else:
                segment_losses = None
            if source.has_centers:
                center_losses = source.calculate_label_metric(transform, target, "centers")
            else:
                center_losses = None
            losses.append(({"file_name": osp.basename(file_path),
                            "slice_start": bounds[0].tolist(),
                            "slice_end":  bounds[1].tolist(),
                            "loss": loss,
                            "transform": str(transform),
                            "fit_persent": len(fit) / len(source.features),
                            "num_of_features": len(source.features),
                            "segment_losses": segment_losses,
                            "center_losses": center_losses
                            }))

            result_dict = {"results": losses
                           }

            o3d.io.write_point_cloud(
                osp.join(save_path, osp.basename(file_path).replace(".ply", f"_{slicer_index}.ply")),
                (transform.transform(source.down_pcd)))
            with open(osp.join(save_path, f"{time_string}_results.json"), "w") as f:
                json.dump(result_dict, f, cls=NumpyEncoder)


if __name__ == "__main__":

    time_string = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    target_path = r"D:\datasets\VerSe2020\new_train\sub-verse506.ply"
    folder_path = r"D:\datasets\VerSe2020\new_train\\"
    figuring_fpfh(target_path=target_path,
              source_folder_path=folder_path,
              save_path=fr"D:\research_results\HyperSkeleton\{time_string}")


