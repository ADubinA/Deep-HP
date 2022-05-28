import copy
import json
from datetime import datetime

import numpy as np
import torch
from torch_geometric.data import Data

from classical_registration.feature_hyper_skeleton import HyperSkeleton
from classical_registration.other_methods import FPFH
from network.deep_utils import RandomCrop, RandomScaleAxis
from network.train import DeepHyperSkeleton, Net
import os, glob
import os.path as osp
import open3d as o3d
from classical_registration.utils import z_percent_slicer, NumpyEncoder
from torch_geometric import transforms as T
import random
import argparse

@torch.no_grad()
def run_results_verse(model_setup, target_path, source_folder_path, save_path, description="", has_labels=True, has_centers=True, slicer=0.5,data_type="verse"):
    os.mkdir(os.path.join(save_path))
    target = model_setup(target_path, has_labels=has_labels, has_centers=has_centers)
    target.create_global_features()
    o3d.io.write_point_cloud(osp.join(save_path, osp.basename(target_path).replace(".ply", "_ref.ply")), target.down_pcd)
    losses = []
    for file_path in glob.glob(osp.join(source_folder_path, "*.ply")):
        if "center" in file_path or "label" in file_path:
            continue
        if data_type not in file_path:
            continue
        print(file_path)
        for bounds in z_percent_slicer(slicer):

            source = model_setup(file_path,bounds=bounds, has_labels=has_labels, has_centers=has_centers, cluster_func=target.cluster_func)
            if len(source.base_pcd.points)<250:
                continue

            source.create_global_features()
            transform, loss, fit = source.register(target)
            if source.has_labels:
                segment_losses = source.calculate_label_metric(transform,target,"segment")
            else:
                segment_losses = None
            if source.has_centers:
                center_losses = source.calculate_label_metric(transform, target, "centers")
            else:
                center_losses = None
            losses.append(({"file_name": osp.basename(file_path),
                            "bounds": bounds,
                            "loss": loss,
                            "transform":str(transform),
                            "fit_percent": len(fit)/len(source.features),
                            "num_of_features": len(source.features),
                            "segment_losses":segment_losses,
                            "center_losses": center_losses,
                            "min_oriented_bounds": target.down_pcd.get_oriented_bounding_box().get_min_bound(),
                            "max_oriented_bounds": target.down_pcd.get_oriented_bounding_box().get_max_bound()
                            }))

            result_dict = {"results":losses, "description":description}
            slice_str = str(bounds.tolist()).replace(",", "_").replace(".", "").replace(" ", "")
            o3d.io.write_point_cloud(osp.join(save_path,osp.basename(file_path).replace(".ply", f"_{slice_str}.ply")),
                                         (transform.transform(source.down_pcd)))
            with open(osp.join(save_path,f"{time_string}_results.json"),"w") as f:
                json.dump(result_dict, f, cls=NumpyEncoder)

@torch.no_grad()
def run_results_same(model_setup, source_folder_path, save_path, description="", has_labels=True, has_centers=True, slicer=0.5,data_type="verse"):
    os.mkdir(os.path.join(save_path))

    losses = []
    for file_path in glob.glob(osp.join(source_folder_path, "*.ply")):
        if "center" in file_path or "label" in file_path:
            continue
        if data_type not in file_path:
            continue
        print(file_path)
        target = model_setup(file_path, has_labels=has_labels, has_centers=has_centers)
        target.create_global_features()
        o3d.io.write_point_cloud(osp.join(save_path, osp.basename(file_path).replace(".ply", "_full.ply")),
                                 target.down_pcd)

        for bounds in z_percent_slicer(slicer):

            source = model_setup(file_path,bounds=bounds, has_labels=has_labels, has_centers=has_centers, cluster_func=target.cluster_func)
            if len(source.base_pcd.points)<250:
                continue
            if AUGMENT:
                augmentation(source)
            source.create_global_features()
            transform, loss, fit = source.register(target)
            if source.has_labels:
                segment_losses = source.calculate_label_metric(transform,target,"segment")
            else:
                segment_losses = None
            if source.has_centers:
                center_losses = source.calculate_label_metric(transform, target, "centers")
            else:
                center_losses = None
            losses.append(({"file_name": osp.basename(file_path),
                            "bounds": bounds,
                            "loss": loss,
                            "transform":str(transform),
                            "fit_percent": len(fit)/len(source.features),
                            "num_of_features": len(source.features),
                            "segment_losses":segment_losses,
                            "center_losses": center_losses,
                            "min_oriented_bounds": target.down_pcd.get_oriented_bounding_box().get_min_bound(),
                            "max_oriented_bounds": target.down_pcd.get_oriented_bounding_box().get_max_bound()

                            }))

            result_dict = {"results":losses, "description":description}
            slice_str = str(bounds.tolist()).replace(",", "_").replace(".", "").replace(" ", "")
            o3d.io.write_point_cloud(osp.join(save_path,osp.basename(file_path).replace(".ply", f"_{slice_str}.ply")),
                                         (transform.transform(source.down_pcd)))
            with open(osp.join(save_path,f"{time_string}_results.json"),"w") as f:
                json.dump(result_dict, f, cls=NumpyEncoder)


@torch.no_grad()
def run_results_noise(model_setup, source_folder_path, save_path, description="", has_labels=True, has_centers=True, slicer=0.5,data_type="verse", noise=0.005):
    os.mkdir(os.path.join(save_path))

    losses = []
    for file_path in glob.glob(osp.join(source_folder_path, "*.ply")):
        if "center" in file_path or "label" in file_path:
            continue
        if data_type not in file_path:
            continue
        print(file_path)
        target = model_setup(file_path, has_labels=has_labels, has_centers=has_centers)
        target.create_global_features()
        o3d.io.write_point_cloud(osp.join(save_path, osp.basename(file_path).replace(".ply", "_full.ply")),
                                 target.down_pcd)
        for bounds in z_percent_slicer(slicer):

            source = model_setup(file_path,bounds=bounds, has_labels=has_labels, has_centers=has_centers, cluster_func=target.cluster_func)
            if len(source.base_pcd.points)<250:
                continue
            augmentation(source, noise_percent=noise, added_scale=0, added_rotation=0)
            source.create_global_features()
            transform, loss, fit = source.register(target)
            if source.has_labels:
                segment_losses = source.calculate_label_metric(transform,target,"segment")
            else:
                segment_losses = None
            if source.has_centers:
                center_losses = source.calculate_label_metric(transform, target, "centers")
            else:
                center_losses = None
            losses.append(({"file_name": osp.basename(file_path),
                            "bounds": bounds,
                            "loss": loss,
                            "transform":str(transform),
                            "fit_percent": len(fit)/len(source.features),
                            "num_of_features": len(source.features),
                            "segment_losses":segment_losses,
                            "center_losses": center_losses,
                            "min_oriented_bounds": target.down_pcd.get_oriented_bounding_box().get_min_bound(),
                            "max_oriented_bounds": target.down_pcd.get_oriented_bounding_box().get_max_bound()
                            }))

            result_dict = {"results":losses, "description":description}
            slice_str = str(bounds.tolist()).replace(",", "_").replace(".", "").replace(" ", "")
            o3d.io.write_point_cloud(osp.join(save_path,osp.basename(file_path).replace(".ply", f"_{slice_str}.ply")),
                                         (transform.transform(source.down_pcd)))
            with open(osp.join(save_path,f"{time_string}_results.json"),"w") as f:
                json.dump(result_dict, f, cls=NumpyEncoder)

def deep_hp_model_setup(pcd_path, bounds= np.array([[0,0,0],[1,1,1]]),has_labels=True, has_centers = True, cluster_func=None):

    reg_model = DeepHyperSkeleton(pcd_path, model, bounds=bounds, has_labels=has_labels, has_centers=has_centers)
    reg_model.voxel_down_sample_size = 2
    reg_model.graph_point_distance = 2.5
    reg_model.load_pcd()
    return reg_model
def hp_model_setup(pcd_path, bounds= np.array([[0,0,0],[1,1,1]]),has_labels=True, has_centers = True, cluster_func=None):
    reg_model = HyperSkeleton(pcd_path,bounds = bounds, has_labels=has_labels, has_centers=has_centers, cluster_func=cluster_func)
    reg_model.voxel_down_sample_size = 4
    reg_model.graph_point_distance = 4.5
    reg_model.load_pcd()
    return reg_model

def fpfh_model_setup(pcd_path, bounds= np.array([[0,0,0],[1,1,1]]),has_labels=True, has_centers = True, cluster_func=None):
    reg_model = FPFH(pcd_path, bounds, has_labels=has_labels, has_centers=has_centers)
    reg_model.voxel_down_sample_size = 8
    reg_model.load_pcd()
    return reg_model

def augmentation(feature_reg, noise_percent = 0.002, added_rotation=0, added_scale=0.1):
    bounds = feature_reg.base_pcd.get_oriented_bounding_box()
    diag_len = np.linalg.norm(bounds.get_max_bound() - bounds.get_min_bound())
    pcd_list = [feature_reg.base_pcd, feature_reg.down_pcd]
    if feature_reg.has_labels:
        pcd_list.append(feature_reg.segments)
    if feature_reg.has_centers:
        pcd_list.append(feature_reg.centers)

    random_scale = np.diag(np.append(np.random.uniform(1-added_scale, 1+added_scale, 3),[1]))
    random_rotation = np.random.uniform(-added_rotation,added_rotation,3)
    for pcd in pcd_list:
        pcd.transform(random_scale)
        pcd.rotate(pcd.get_rotation_matrix_from_axis_angle(random_rotation))

    feature_reg.down_pcd.points = o3d.utility.Vector3dVector(
        np.asarray(feature_reg.down_pcd.points)+np.random.normal(scale=diag_len*noise_percent,
                                                                 size = np.asarray(feature_reg.down_pcd.points).shape))

def view_results(json_path, show_sub_loss=False,percent=True):
    with open(json_path, "r") as f:
        results = json.load(f)
    print(f"Description: \n{results['description']}------------------------------------------------")
    results = results["results"]
    consensus_percent = np.array([result.get("fit_percent",result.get("fit_persent",0)) for result in results])
    not_outliars = np.array([result["loss"] for result in results if result["loss"] > 0 and result["loss"] < 100])
    if percent:
        sizes = np.array([calculate_size_for_results(result) for result in results])
    else:
        sizes = np.array([1])

    not_outliars = not_outliars/sizes*100
    print(f"mean results: {not_outliars.mean():.2f} \\pm {not_outliars.std():.2f}")
    # print(f"Non detection rate is {len([result for result in results if result['loss']>100 or result['loss']<0])/len(results)}")

    segment_loss = {}
    center_loss = {}
    for i, result in enumerate(results):
        if result.get("segment_losses"):
            for key, loss in result["segment_losses"].items():
                if loss == -1:
                    continue
                segment_loss[key] = segment_loss.get(key, []) + [loss/sizes[i]*100]
        if result.get("center_losses"):
            for key, loss in result["center_losses"].items():
                if loss == -1:
                    continue
                center_loss[key] = center_loss.get(key, []) + [loss/sizes[i]*100]

    segment_total_loss = []
    for key, losses in segment_loss.items():
        segment_total_loss.extend(losses)
        if show_sub_loss:
            print(f"{key}: has {len(losses)} items and mean: {np.array(losses).mean():.2f}, std {np.array(losses).std():.2f}")
    segment_total_loss = np.array(segment_total_loss)
    print(f"total segment losses mean: {segment_total_loss.mean():.2f} \\pm {segment_total_loss.std():.2f}")
    center_total_loss = []
    for key, losses in center_loss.items():
        center_total_loss.extend(losses)
        if show_sub_loss:
            print(f"{key}: has {len(losses)} items and mean: {np.array(losses).mean():.2f}, std {np.array(losses).std():.2f}")
    center_total_loss = np.array(center_total_loss)
    print(f"total center losses mean: {center_total_loss.mean():.2f} \\pm {center_total_loss.std():.2f}")
    print(f"consensus percent: {consensus_percent.mean():.2f} \\pm {consensus_percent.std():.2f}")

def calculate_size_for_results(result):
    return np.linalg.norm(np.array(result.get("max_oriented_bounds")) - np.array(result.get("min_oriented_bounds")))

def full_summery(folder_path):
    for jfile_path in glob.glob(osp.join(folder_path, "*", "*_results.json")):
        view_results(jfile_path)

if __name__ == "__main__":

    # result_path = r"D:\research_results\HyperSkeleton\new_article_results"
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    # full_summery(result_path)
    seed = 8008135
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    time_string = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 64
    model_path = r"F:\dev\pointermorpher\network\runs\03-14-2022_07-57-15\best_model.pt"
    model = Net(num_classes).to(device)
    model.load_state_dict(torch.load(model_path))

    parser = argparse.ArgumentParser(description='testing deep, hp or fpfh')
    parser.add_argument('--algo_type', required=True, choices=['hp', 'deephp', 'fpfh'])
    parser.add_argument('--aug', action='store_true', help='use augmentation or not')
    parser.add_argument('--exp_type', required=True, choices=['same', 'verse', 'gl'])

    args = parser.parse_args()
    AUGMENT = args.aug
    setup_model = {"hp":hp_model_setup, "deephp":deep_hp_model_setup, "fpfh":fpfh_model_setup}[args.algo_type]
    try:

        if args.exp_type == "same":
            run_results_same(setup_model,
                              r"D:\datasets\VerSe2020\new_validation",
                              fr"D:\research_results\HyperSkeleton\{time_string}",
                              description= f"{args.algo_type} with 0.5 zslicer with validation of {args.exp_type} and has {args.aug} noise",
                              slicer=0.5, data_type="verse")
        elif args.exp_type == "verse":
            run_results_verse(setup_model, r"D:\datasets\VerSe2020\new_validation\sub-verse508.ply",
                              r"D:\datasets\VerSe2020\new_validation",
                              fr"D:\research_results\HyperSkeleton\{time_string}",
                              description= f"{args.algo_type} with 0.5 zslicer with validation of {args.exp_type} and has {args.aug} noise",
                              slicer=0.5, data_type="verse")
        elif args.exp_type == "gl":
            run_results_verse(setup_model, r"D:\datasets\VerSe2020\new_validation\sub-gl045.ply",
                              r"D:\datasets\VerSe2020\new_validation",
                              fr"D:\research_results\HyperSkeleton\{time_string}",
                              description= f"{args.algo_type} with 0.5 zslicer with validation of {args.exp_type} and has {args.aug} noise",
                              slicer=0.5, data_type=args.exp_type)
    except Exception as e:
        save_path = fr"D:\research_results\HyperSkeleton\{time_string}"
        text = f"{args.algo_type} with 0.5 zslicer with validation of {args.exp_type} and has {args.aug} noise"
        print(f"bad results of {text}")
        with open(osp.join(save_path, f"{time_string}_results.json"), "w") as f:

            json.dump({"error":str(e), "description": text},f)
