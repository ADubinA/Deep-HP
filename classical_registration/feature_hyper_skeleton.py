import copy
import glob
import json
import os
import random
import datetime

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import open3d as o3d
from tqdm import tqdm
import tps
from classical_registration import utils
from classical_registration.create_atlas import to_spherical
from sklearn.cluster import KMeans

from classical_registration.utils import chamfer_distance, calculate_curvature,NumpyEncoder, better_graph_contraction
from scipy.spatial import KDTree

class FeaturedPointCloud:
    def __init__(self):
        super(FeaturedPointCloud, self).__init__()
        self.features = []
        self.pcd = o3d.geometry.PointCloud()
        self.n_iter = None

    @property
    def points(self):
        return self.pcd.points
    @points.setter
    def points(self, points):
        self.pcd.points = points

    def transform(self, trans):
        return self.pcd.transform(trans)
    def append(self, data):
        self.pcd.points.append(data.pop('mean'))
        self.features.append(data)

        color = data.get("color", None)
        if color is not None:
            self.pcd.colors.append(color)

    def __iter__(self):
        self.n_iter = 0
        return self

    def __len__(self):
        return len(self.pcd.points)

    def __next__(self):
        if self.n_iter < len(self):
            iter_dict = self[self.n_iter]
            self.n_iter += 1
            return iter_dict
        else:
            raise StopIteration
    def __getitem__(self, item):
        iter_dict = {'mean': np.asarray(self.pcd.points)[item], "index": item}
        iter_dict.update(self.features[item])
        return iter_dict

class ABSFeatures:
    def __init__(self, path, bounds=(np.array([0, 0, 0]), np.array([1, 1, 1])),
                 voxel_down_sample_size=4, has_labels=False,has_centers=False):
        self.path = path
        self.voxel_down_sample_size = voxel_down_sample_size

        # data
        self.base_pcd = None
        self.down_pcd = None
        self.features = None
        self.segments = None
        self.centers = None

        self.bounds = bounds
        self.has_labels = has_labels
        self.has_centers = has_centers
        self.registration_alg = ABSRegistration()
        self.base_translate = None


    def load_pcd(self):
        self.base_pcd = o3d.io.read_point_cloud(self.path)
        # translate minimum of the pcd to (0,0,0)
        self.base_translate = -self.base_pcd.get_min_bound()

        self.down_pcd = self._preprocess_pcd(self.base_pcd, True)
        self.down_pcd.paint_uniform_color([0.1, 0.1, 0.1])

        if self.has_labels:
            self.segments = o3d.io.read_point_cloud(self.path.replace('.', '_labels.'))
            self.segments = self._preprocess_pcd(self.segments)
        if self.has_centers:
            self.centers = o3d.io.read_point_cloud(self.path.replace('.', '_centers.'))
            self.centers = self._preprocess_pcd(self.centers)
    def augment(self, noise_std, ):
        points = np.asarray(self.down_pcd.points)
        points = points + np.random.normal(loc=0, scale=noise_std,size=points.shape)
        self.down_pcd.points = o3d.utility.Vector3dVector(points)
    def _preprocess_pcd(self, pcd, downsample=False):
        pcd.translate(self.base_translate)

        numpy_pcd = np.asarray(pcd.points)
        min_bounds = pcd.get_min_bound() + self.bounds[0] * (pcd.get_max_bound() - pcd.get_min_bound())
        max_bounds = pcd.get_min_bound() + self.bounds[1] * (pcd.get_max_bound() - pcd.get_min_bound())
        idx = np.logical_and(np.all(numpy_pcd > min_bounds, axis=1),
                             np.all(numpy_pcd < max_bounds, axis=1))
        if pcd.has_colors():
            pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[idx])
            pcd.points = o3d.utility.Vector3dVector(numpy_pcd[idx])
        else:
            pcd.points = o3d.utility.Vector3dVector(numpy_pcd[idx])

        if downsample:
            down_pcd = pcd.voxel_down_sample(self.voxel_down_sample_size)
        else:
            down_pcd = pcd
        return down_pcd

    def calculate_label_metric(self, transform, target, label_type):
        if label_type == "segment":
            source_pcd = self.segments
            target_pcd = target.segments
        else:
            source_pcd = self.centers
            target_pcd = target.centers

        moved = transform.transform(copy.deepcopy(source_pcd))
        unique_labels = np.unique(np.asarray(source_pcd.colors))
        results = {}
        for unique_label in unique_labels:
            dist = chamfer_distance(np.asarray(moved.points)[np.asarray(moved.colors)[:,0] == unique_label],
                             np.asarray(target_pcd.points)[np.asarray(target_pcd.colors)[:,0] == unique_label],
                             direction="x_to_y")

            results[unique_label] = dist
        return results

    def visualize_results(self, consensus, target):
        pcd_transform = self.registration_alg.get_transform(consensus)
        moved = pcd_transform.transform(copy.deepcopy(self.down_pcd))

        lineset = o3d.geometry.LineSet()
        lines, connections, colors = [], [], []
        for idx, line in enumerate(consensus):
            lines.extend([line[0]["mean"], line[1]["mean"]])
            connections.append([2*idx, 2*idx+1])
            colors.append(np.random.random(3))

        lineset.points = o3d.utility.Vector3dVector(lines)
        lineset.lines = o3d.utility.Vector2iVector(connections)
        lineset.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw([lineset, moved,target.down_pcd, target.features.pcd, self.down_pcd, self.features.pcd])

class HyperSkeleton(ABSFeatures):
    def __init__(self, path, bounds=(np.array([0,0,0]), np.array([1,1,1])),
                 voxel_down_sample_size=2, cluster_func=None, cluster_colors=None, has_labels=False,
                 has_centers=False):
        super().__init__(path, bounds,voxel_down_sample_size, has_labels, has_centers)
        self.graph_point_distance = 2.5
        self.voxel_down_sample_size = voxel_down_sample_size
        self.min_path_lengths = 15
        self.max_path_lengths = 20

        # clustering
        self.num_bins = 8
        self.n_clusters = 64
        self.minimum_cluster_eig = 2
        self.path = path
        self.cluster_func = cluster_func
        if cluster_colors is None:
            self.cluster_colors = np.random.random((self.n_clusters,3))
        else:
            self.cluster_colors = cluster_colors

    def save_features(self, folder_path):
        base_name = os.path.basename(self.path)
        _, bins_list = self._create_local_features(self.down_pcd)
        o3d.io.write_point_cloud(os.path.join(folder_path, base_name), self.down_pcd)
        np.save(os.path.join(folder_path, base_name.replace(".ply", "_hist.npy")), bins_list)

    def create_global_features(self):
        g, bins_list = self._create_local_features(self.down_pcd)
        g = self._cluster(g, bins_list)
        self.create_featured_pcd(g)

    def create_featured_pcd(self, g):
        self.features = FeaturedPointCloud()
        for connected in nx.connected_components(g):
            if len(list(connected)) < 3:
                continue

            cluster_label = g.nodes[list(connected)[0]]["label"]
            cluster_color = g.nodes[list(connected)[0]]["color"]
            points = np.stack([g.nodes[node]["point"] for node in connected])
            cov = np.cov(points.T)
            eig = np.linalg.eig(cov)[0]
            if np.sqrt(np.max(eig)) < self.minimum_cluster_eig:
                continue

            # if self.features.get(cluster_label,None) is None:
            #     self.features[cluster_label] = []

            self.features.append({"mean": points.mean(axis=0),
                                  "cov": cov,
                                  "eig": eig,
                                  "label": cluster_label,
                                  "color": cluster_color,
                                  "cluster": np.array(list(connected))})

    def create_graph(self, pcd):
        g = nx.Graph()
        for point_index in range(np.asarray(pcd.points).shape[0]):
            g.add_node(point_index, point=np.asarray(pcd.points)[point_index], index=point_index)

        bone_tree = o3d.geometry.KDTreeFlann(pcd)
        for point_index in range(np.asarray(pcd.points).shape[0]):
            [k, nie, _] = bone_tree.search_radius_vector_3d(pcd.points[point_index], self.graph_point_distance)

            for element in nie:
                if element == point_index:
                    continue
                g.add_edge(point_index, element,
                           weight=np.linalg.norm(np.asarray(pcd.points)[point_index] - np.asarray(pcd.points)[element]))

        return g

    def _create_local_features(self, pcd):

        g = self.create_graph(pcd)
        # g = better_graph_contraction(g,int(g.number_of_nodes()/2), 6)
        dix = nx.all_pairs_shortest_path(g, self.max_path_lengths)
        # dix = networkx.all_pairs_dijkstra(g, max_path_lengths)
        bins_list = []
        for point_index, paths in tqdm(dix, total=g.number_of_nodes(), desc="Creating local features"):
            # paths = paths[1]  # get only the paths, not the lens
            paths = [path for _, path in paths.items() if len(path) > self.min_path_lengths]
            if len(paths) < 5:
                bins = np.zeros((self.num_bins, self.num_bins))
            else:
                path_edges = [g.nodes[path[-1]]["point"] for path in paths]
                dist = np.asarray(path_edges) - g.nodes[point_index]["point"]
                dist = to_spherical(dist)
                bins, _, _ = np.histogram2d(dist[:, 1], dist[:, 2], np.arange(self.num_bins + 1) * 1 / self.num_bins,
                                            density=True, range=(0, 1))

                bins = bins / bins.sum()
            g.nodes[point_index]["histogram"] = bins
            bins_list.append(bins)

        bins_list = np.stack(bins_list)
        return g, bins_list

    def _cluster(self, g, bins_list):
        flat_bins = bins_list.reshape(bins_list.shape[0], -1)
        if self.cluster_func is None:
            self.cluster_func = KMeans(init="k-means++", n_clusters=self.n_clusters, n_init=5)
            results = self.cluster_func.fit_predict(flat_bins)
        else:
            results = self.cluster_func.predict(flat_bins)

        # histogram mapping colors
        np.asarray(self.down_pcd.colors)[np.array(list(g.nodes))] = self.cluster_colors[results]

        for node in g.nodes:
            g.nodes[node]['label'] = results[node]
            g.nodes[node]['color'] = self.cluster_colors[results[node]]
        different_cluster_edges = [edge for edge in g.edges if g.nodes[edge[0]]["label"] != g.nodes[edge[1]]["label"]]
        g.remove_edges_from(different_cluster_edges)

        return g

    def register(self, target):
        return self.registration_alg.register(self, target)
    # def local_refine_icp(self, correspondences, target, max_distance=1,init=None):
    #     if not init:
    #         init = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    #
    #     total_pcd = o3d.geometry.PointCloud()
    #     for correspondence in correspondences:
    #
    #         source_cluster = np.asarray(self.down_pcd.points)[correspondence[0]["cluster"]]
    #         source_pcd = o3d.geometry.PointCloud()
    #         source_pcd.points = o3d.utility.Vector3dVector(source_cluster)
    #         source_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    #
    #         target_cluster = np.asarray(target.down_pcd.points)[correspondence[1]["cluster"]]
    #         target_pcd = o3d.geometry.PointCloud()
    #         target_pcd.points = o3d.utility.Vector3dVector(target_cluster)
    #         target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    #
    #         source_pcd.translate(
    #             target_pcd.compute_mean_and_covariance()[0] - source_pcd.compute_mean_and_covariance()[0])
    #         reg = o3d.pipelines.registration.registration_icp(source_pcd, target_pcd, max_distance, init,
    #                                                     o3d.pipelines.registration.TransformationEstimationPointToPoint())
    #
    #         source_pcd.transform(reg.transformation)
    #         total_pcd.points.extend(source_pcd.points)
    #     return total_pcd


    @staticmethod


    def visualize_feature_distributions(self):
        feature_dict = {}
        for feature in self.features:
            if not feature_dict.get(feature["label"], False):
                feature_dict[feature["label"]] = []
            feature_dict[feature["label"]].append(np.linalg.det(feature['cov']))

        for label, feature in feature_dict.items():
            plt.hist(feature)


class ABSRegistration:
    def __init__(self, ):
        # registration metric
        self.minimum_feature_distance = 100
        self.minimum_correspondence_distance = 20
        self.min_correspondence_percent = 0.2
        self.finish_fit_percent = 0.9
        self.num_samples = 20

    def get_transform(self, correspondences):
        if not correspondences:
            return np.eye(4)
        X = np.stack([correspondence[0]["mean"] for correspondence in correspondences])
        Y = np.stack([correspondence[1]["mean"] for correspondence in correspondences])

        X = np.insert(X, 3, 1, axis=1)
        Y = np.insert(Y, 3, 1, axis=1)

        X = X.transpose()
        mat = np.matmul(np.linalg.pinv(X).transpose(), Y).transpose()
        return AffinePcdTransform(mat)

    def register(self,source, target, iterations=20, prefer_fit=True):
        # for each label, find all consensuses
        best_fit = []
        best_error = float("infinity")
        pbar = tqdm(range(iterations))
        for i in pbar:
            pbar.set_description(f"Best loss: {best_error:.2f} with fit: {len(best_fit) / len(source.features)}.")
            fit, error = self._registration_iteration(source, target)
            if len(fit) / len(source.features) < self.min_correspondence_percent:
                if len(fit) > len(best_fit):
                    best_fit = fit
                continue
            if prefer_fit:
                if len(fit) / len(source.features) > len(best_fit) / len(source.features):
                    best_fit, best_error = fit, error
            else:
                if best_error > error:
                    best_fit, best_error = fit, error

            if len(best_fit) / len(source.features) > self.finish_fit_percent:
                pbar.set_description(f"Best loss: {best_error:.2f} with fit: {len(best_fit) / len(source.features)}.")
                break

        if best_error == float("infinity"):
            transform = self.get_transform(best_fit)
            moved = transform.transform(copy.deepcopy(source.down_pcd))
            best_error = chamfer_distance(np.asarray(moved.points), np.asarray(target.down_pcd.points), direction="x_to_y")

        # self.visualize_results(best_fit,target)
        return self.get_transform(best_fit), best_error, best_fit

    def _registration_iteration(self, source, target):

        # calculate histogram of differences and take the samples with consensus
        samples = self._get_correspondence(source, target)
        consensus = self._get_sample_concensus(samples)
        # consensus = samples
        if not consensus:
            return [], float("infinity")
        # calculate chamfer distance
        pcd_transform = self.get_transform(consensus)
        full_consensus = self._get_full_correspondence(pcd_transform, source, target)
        full_transform = self.get_transform(full_consensus)
        moved = full_transform.transform(copy.deepcopy(source.down_pcd))
        loss = chamfer_distance(np.asarray(moved.points), np.asarray(target.down_pcd.points), direction="x_to_y")

        return full_consensus, loss

    def _get_full_correspondence(self, transform, source, target):
        # transform the features from this to the target
        transformed_features = copy.deepcopy(source.features)
        transformed_features = transform.transform(transformed_features)
        bone_tree = o3d.geometry.KDTreeFlann(target.features.pcd)
        consensus = []
        for transformed_feature in transformed_features:
            # get all points that are close enough and have the same label
            [k, nie, _] = bone_tree.search_radius_vector_3d(transformed_feature["mean"],
                                                            self.minimum_correspondence_distance)
            target_features = [target.features[i] for i in nie if
                               target.features[i]["label"] == transformed_feature["label"]]
            if not len(target_features):
                continue

            # get the closest point
            # match = max(target_features, key=lambda x: utils.gaussian_wasserstein_dist(
            #     x["mean"], transformed_feature["mean"], x["cov"], transformed_feature["cov"]))
            match = max(target_features, key=lambda x:np.linalg.norm(x["mean"] - transformed_feature["mean"]))
            consensus.append((source.features[transformed_feature["index"]], match))

        return consensus

        # for every feature that doesn't have a match
        # find a match that is of the same class
        # and is not far from it
        # and has similar std

    def _get_correspondence(self, source,  target):
        correspondence = []
        for sample in source.features:
            # try to find a sample in target that is not  in samples that matches
            target_options = []
            for target_feature in target.features:
                # if np.isin(target_feature, [sample[1] for sample in samples]):  # was not picked already
                #     continue
                if target_feature["label"] != sample["label"]:  # needs to have same label
                    continue
                # if utils.gaussian_wasserstein_dist(0, 0, sample["cov"],
                #                                    target_feature["cov"]) > self.minimum_feature_distance:  # is close
                #     continue
                target_options.append(target_feature)

            if len(target_options) == 0:
                continue

            random_target_option = random.choice(target_options)
            correspondence.append((sample, random_target_option))

        return correspondence

    def _get_sample_concensus(self, samples):
        bins = 5
        # calculate differences
        samples_differences = np.stack([sample[0]["mean"] - sample[1]["mean"] for sample in samples])

        # get histogram, calculate the max bin
        hist, edges = np.histogramdd(samples_differences, bins)
        if hist.max() < 3:
            return []
        max_edges = np.unravel_index(hist.argmax(), hist.shape)

        # get the features in that bin
        consensus = []
        for sample_index in range(len(samples_differences)):
            sample_difference = samples_differences[sample_index]
            for axis in range(3):
                if sample_difference[axis] < edges[axis][max_edges[axis]] \
                        or sample_difference[axis] > edges[axis][max_edges[axis] + 1]:
                    axis = -1
                    break
            if axis != -1:
                consensus.append(samples[sample_index])

        return consensus

class TPSRegistration(ABSRegistration):
    def get_transform(self, correspondences):
        X = np.stack([correspondence[0]["mean"] for correspondence in correspondences])
        Y = np.stack([correspondence[1]["mean"] for correspondence in correspondences])
        trans = tps.TPS(X, Y)
        return TPSPcdTransform(trans)


class ABSPcdTransform:
    def __init__(self, transform):
        self.transform_params = transform

    def __str__(self):
        return str(self.transform_params)

    def transform(self, pcd):
        raise NotImplemented()

class AffinePcdTransform(ABSPcdTransform):
    def transform(self, pcd):
        pcd.transform(self.transform_params)
        return pcd


class TPSPcdTransform(ABSPcdTransform):
    def transform(self, pcd):
        pcd.points = o3d.utility.Vector3dVector(self.transform_params(np.asarray((pcd.points))))
        return pcd
    def __str__(self):
        return str(self.transform_params.coefficient)

class HyperSkeletonCurve(HyperSkeleton):
    def __init__(self, path, min_z=150, max_z=200, cluster_func=None,cluster_colors=None):

        super().__init__(path, min_z=min_z, max_z=max_z, cluster_func=None,cluster_colors=None)

        self.num_of_paths = 20  # number of paths per points to sample curvature
        self.histogram_range = (0,0.5)
        self.n_clusters = 8
    def _create_local_features(self, pcd):

        g = self._create_graph(pcd)
        # for _ in range(3):
        #     g = better_graph_contraction(g, int(g.number_of_nodes() / 2), 5)

        dix = nx.all_pairs_shortest_path(g, self.max_path_lengths)
        # dix = networkx.all_pairs_dijkstra(g, max_path_lengths)
        bins_list = []
        for point_index, paths in tqdm(dix, total=g.number_of_nodes(), desc="Creating local features"):
            # paths = paths[1]  # get only the paths, not the lens
            paths = [path for _, path in paths.items() if len(path) == self.max_path_lengths]
            if len(paths) < self.num_of_paths:
                bins = np.zeros(self.num_bins)
            else:
                chosen_paths = random.sample(paths, self.num_of_paths)
                rads = []
                for path in chosen_paths:
                    path_points = np.array([g.nodes[i]["point"] for i in path])
                    rads.append(calculate_curvature(path_points))

                bins, _ = np.histogram(rads,self.num_bins, density=True, range=self.histogram_range)
            g.nodes[point_index]["histogram"] = bins
            bins_list.append(bins)

        bins_list = np.stack(bins_list)
        return g, bins_list


def test_HyperSkeleton(target_path,source_folder_path,save_path, description=""):
    target = HyperSkeleton(target_path, min_z=0, max_z=1000, has_labels=True,has_centers=True)
    target.create_global_features()

    subvolume_size = 100
    losses = []
    for file_path in glob.glob(os.path.join(source_folder_path,"*.ply")):
        if "center" in file_path or "label" in file_path or "gl" in file_path:
            continue
        print(file_path)
        for slicer_index in [100,200,300,400]:

            source = HyperSkeleton(file_path,
                                   min_z=slicer_index, max_z=slicer_index+subvolume_size,
                                   cluster_func=target.cluster_func, cluster_colors=target.cluster_colors, has_labels=True,has_centers=True)
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
            losses.append(({"file_name": os.path.basename(file_path),
                            "slice_start": slicer_index,
                            "slice_end": slicer_index+subvolume_size,
                            "loss": loss,
                            "transform":str(transform),
                            "fit_persent": len(fit)/len(source.features),
                            "num_of_features": len(source.features),
                            "segment_losses": segment_losses,
                            "center_losses": center_losses
                            }))

            result_dict = {"results": losses, "description": description,
                           "minimum_cluster_eig": source.minimum_cluster_eig,
                           "minimum_feature_distances": source.registration_alg.minimum_feature_distance,
                           "min_correspondence_percent": source.registration_alg.min_correspondence_percent,
                           "n_clusters":source.n_clusters,
                           "num_bins":source.num_bins,
                           "finish_fit_percent":source.registration_alg.finish_fit_percent,
                           "graph_point_distance":source.graph_point_distance,
                           "max_path_lengths":source.max_path_lengths,
                           "minimum_correspondence_distance": source.registration_alg.minimum_correspondence_distance
                           }
            with open(save_path,"w") as f:
                json.dump(result_dict, f, cls=NumpyEncoder)

def test_view_results(json_path):
    with open(json_path, "r") as f:
        results = json.load(f)
    print(f"Description: \n{results['description']}")
    results = results["results"]
    not_outliars = np.array([result["loss"] for result in results if result["loss"] > 0 and result["loss"] < 100])
    print(f"mean results: {not_outliars.mean()} with std of {not_outliars.std()}")
    print(f"Non detection rate is {len([result for result in results if result['loss']>100 or result['loss']<0])/len(results)}")
    print("------------------------------------")
    for i in [0,100,200,300,400]:
        slice_results = [result["loss"] for result in results if result["slice_start"] == i]
        if len(slice_results)==0:
            continue
        not_outliars = np.array([result for result in slice_results if result > 0 and result < 100])
        print(f"for slice {i}: ")
        print(f"mean results: {not_outliars.mean()} with std of {not_outliars.std()}")
        print( f"Non detection rate is {len([result for result in slice_results if result > 100 or result < 0]) / len(slice_results)}")

    for result in results:
        if result["loss"]==-1:
            print(f"{result['file_name']} didn't match at at slice: {result['slice_start']}")
        elif result["loss"]>100:
            print(f"{result['file_name']} had bad loss of {result['file_name']} at slice: {result['slice_start']}")

    segment_loss = {}
    center_loss = {}
    for result in results:
        if result.get("segment_losses"):
            for key, loss in result["segment_losses"].items():
                segment_loss[key] = segment_loss.get(key, []) + [loss]

        if result.get("center_losses"):
            for key, loss in result["center_losses"].items():
                center_loss[key] = center_loss.get(key, []) + [loss]
    print("-------segment losses --------------")
    for key, losses in segment_loss.items():
        print(f"{key}: has {len(losses)} items and mean: {np.array(losses).mean()}, std {np.array(losses).std()}")
    print("-------center losses----------------")
    for key, losses in segment_loss.items():
        print(f"{key}: has {len(losses)} items and mean: {np.array(losses).mean()}, std {np.array(losses).std()}")


def create_training(source_folder_path, output_folder):
    for file_path in glob.glob(os.path.join(source_folder_path, "*.ply")):
        base_name = os.path.basename(file_path)
        if "label" in base_name or "center" in base_name:
            continue
        print(base_name)
        if os.path.exists(os.path.join(output_folder, base_name)):
            continue
        source = HyperSkeleton(file_path,
                               min_z=-10000, max_z=10000)
        source.save_features(output_folder)



if __name__ == "__main__":
    #TODO
    # Remove small clustering of points, and connect them to bigger clusters
    # why is there a sqrt prob
    # figure out how to make the features better in the spine

    # target = HyperSkeleton(r"D:\datasets\VerSe2020\train\sub-verse500.ply", min_z=0, max_z=1000)#, has_labels=True, has_centers=True)
    # target.down_pcd.translate(np.array([500,0,0]))
    # target.create_global_features()
    # o3d.visualization.draw([target.down_pcd, target.features.pcd])
    #
    # source = HyperSkeleton(r"D:\datasets\nmdid\clean-body-pcd\case-100114_BONE_TORSO_3_X_3.ply",
    #                        min_z=150, max_z=300, cluster_func=target.cluster_func,
    #                        cluster_colors=target.cluster_colors)#, has_labels=True, has_centers=True)

    # source = HyperSkeleton(r"D:\datasets\nmdid\clean-body-pcd\case-121936_BONE_TORSO_3_X_3.ply",
    #                        min_z=150, max_z=250, cluster_func=target.cluster_func,
    #                        cluster_colors=target.cluster_colors)
    # source.create_global_features()
    # affine, error, corr = source.register(target)
    # source.visualize_results(corr,target)
    # source.local_refine_icp(corr, target)
    #
    # st = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # test_HyperSkeleton(r"D:\datasets\VerSe2020\train\sub-verse823.ply",
    #                    r"D:\datasets\VerSe2020\*\\",
    #                    fr"D:\research_results\HyperSkeleton\{st}_results.json",
    #                    description= "testing Verse2020 dataset with with segmentation loss")
    results = r"D:\research_results\HyperSkeleton\02-27-2022_22-20-01\02-27-2022_22-20-01_results.json"
    test_view_results(results)

    # create_training(r"D:\datasets\VerSe2020\new_train", r"D:\datasets\VerSe2020\train_hp_labeled\raw")
    # create_training(r"D:\datasets\VerSe2020\new_validation", r"D:\datasets\VerSe2020\validation_hp_labeled\raw")