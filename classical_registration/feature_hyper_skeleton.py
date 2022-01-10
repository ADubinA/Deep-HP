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
from create_atlas import to_spherical
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

class FeaturedPointCloud:
    def __init__(self):
        super(FeaturedPointCloud, self).__init__()
        self.features = []
        self.pcd = o3d.geometry.PointCloud()
        self.n_iter = None
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

class HyperSkeleton:
    def __init__(self, path, min_z=150, max_z=200, cluster_func=None,cluster_colors=None):
        self.graph_point_distance = 4
        self.min_path_lengths = 5
        self.max_path_lengths = 20
        self.num_bins = 8
        self.n_clusters = 128
        self.minimum_cluster_std_mean = 2
        self.min_z, self.max_z = min_z, max_z
        self.path = path
        self.cluster_func = cluster_func
        self.cluster_colors = cluster_colors

        self.base_pcd = None
        self.down_pcd = None
        self.features = None

        # registration metric
        self.minimum_std_distance = 100
        self.minimum_feature_distance = 12
        self.min_correspondence_percent = 0.05
        self.finish_fit_percent = 0.2
        self.num_samples = 20


        self.load_pcd()

    def load_pcd(self):
        self.base_pcd = o3d.io.read_point_cloud(self.path)
        numpy_base_pcd = np.asarray(self.base_pcd.points)
        numpy_base_pcd = numpy_base_pcd[numpy_base_pcd[:, 2] > self.min_z]
        numpy_base_pcd = numpy_base_pcd[numpy_base_pcd[:, 2] < self.max_z]
        self.base_pcd.points = o3d.utility.Vector3dVector(numpy_base_pcd)
        self.down_pcd, _, _ = self.base_pcd.voxel_down_sample_and_trace(self.graph_point_distance,
                                                                      min_bound=np.array([0, 0, self.min_z]),
                                                                      max_bound=np.array([1000, 1000, self.max_z]))
        self.down_pcd.paint_uniform_color([0.1, 0.1, 0.1])

    def create_global_features(self):
        g, bins_list = self._create_local_features(self.down_pcd)
        g = self._cluster(g, bins_list)

        self.features = FeaturedPointCloud()
        for connected in nx.connected_components(g):
            cluster_label = g.nodes[list(connected)[0]]["label"]
            cluster_color = g.nodes[list(connected)[0]]["color"]
            points = np.stack([g.nodes[node]["point"] for node in connected])
            if points.std(axis=0).mean() < self.minimum_cluster_std_mean:
                continue

            # if self.features.get(cluster_label,None) is None:
            #     self.features[cluster_label] = []

            self.features.append({"mean": points.mean(axis=0),
                                   "std": points.std(axis=0),
                                   "label": cluster_label,
                                  "color": cluster_color})

    def _create_graph(self, pcd):
        g = nx.Graph()
        for point_index in range(np.asarray(pcd.points).shape[0]):
            g.add_node(point_index, point=np.asarray(pcd.points)[point_index])

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

        g = self._create_graph(pcd)
        # g = graph_contraction(g,0.5)
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
        if self.cluster_colors is None:
            self.cluster_colors = np.random.random((self.n_clusters,3))
        np.asarray(self.down_pcd.colors)[np.array(list(g.nodes))] = self.cluster_colors[results]

        for node in g.nodes:
            g.nodes[node]['label'] = results[node]
            g.nodes[node]['color'] = self.cluster_colors[results[node]]
        different_cluster_edges = [edge for edge in g.edges if g.nodes[edge[0]]["label"] != g.nodes[edge[1]]["label"]]
        g.remove_edges_from(different_cluster_edges)

        return g

    def get_affine_transform(self, correspondences):
        X = np.stack([correspondence[0]["mean"] for correspondence in correspondences])
        Y = np.stack([correspondence[1]["mean"] for correspondence in correspondences])

        X = np.insert(X,3,1,axis=1)
        Y = np.insert(Y,3,1,axis=1)

        X = X.transpose()
        return np.matmul(np.linalg.pinv(X).transpose(),Y).transpose()

    def register(self, target, iterations=150):
        # for each label, find all consensuses
        best_fit = []
        best_error = float("infinity")
        pbar = tqdm(range(iterations))
        for i in pbar:
            pbar.set_description(f"Best loss: {best_error:.2f} with fit: {len(best_fit)/len(self.features)}.")
            fit, error = self._registration_iteration(target)
            if len(fit)/len(self.features) < self.min_correspondence_percent:
                if len(fit)> len(best_fit):
                    best_fit = fit
                continue
            if best_error > error:
                best_fit,best_error = fit, error

            if len(best_fit)/len(self.features) > self.finish_fit_percent:
                pbar.set_description(f"Best loss: {best_error:.2f} with fit: {len(best_fit) / len(self.features)}.")
                break

        if best_error == float("infinity"):
            return -1,-1


        # self.visualize_results(best_fit,target)
        return self.get_affine_transform(best_fit), best_error

    def _registration_iteration(self,target):

        # random sample points and targets of same size and class
        # samples = []
        # for i in range(self.num_samples):
        #     try:
        #         samples.append(self._sample_matches(target, samples))
        #     except IndexError:
        #         print(f"Warning: not enough features of samples for matching {self.path} and {target.path}.")
        #         break
        # calculate histogram of differences and take the samples with consensus
        samples = self._get_correspondence(target)
        consensus = self._get_sample_concensus(samples)
        # consensus = samples
        if not consensus:
            return [], float("infinity")
        # calculate chamfer distance
        affine_transform = self.get_affine_transform(consensus)
        full_consensus = self._get_full_correspondence(affine_transform,target)
        full_affine_transform = self.get_affine_transform(full_consensus)
        moved = copy.deepcopy(self.down_pcd).transform(full_affine_transform)
        loss = chamfer_distance(np.asarray(moved.points), np.asarray(target.down_pcd.points), direction="x_to_y")

        return full_consensus, loss

    def _get_full_correspondence(self, transform,  target):
        # transform the features from this to the target
        transformed_features = copy.deepcopy(self.features)
        transformed_features.pcd.transform(transform)
        bone_tree = o3d.geometry.KDTreeFlann(target.features.pcd)
        consensus = []
        for transformed_feature in transformed_features:
            [k, nie, _] = bone_tree.search_radius_vector_3d(transformed_feature["mean"],self.minimum_feature_distance)
            target_features = [target.features[i] for i in nie if target.features[i]["label"] ==transformed_feature["label"]]
            if not len(target_features):
                continue

            match = max(target_features, key=lambda x: np.linalg.norm(x["mean"]-transformed_feature["mean"]))
            consensus.append((self.features[transformed_feature["index"]], match))

        return consensus

        # for every feature that doesn't have a match
        # find a match that is of the same class
        # and is not far from it
        # and has similar std

    def _get_correspondence(self, target):
        correspondence = []
        for sample in self.features:
            # try to find a sample in target that is not  in samples that matches
            target_options = []
            for target_feature in target.features:
                # if np.isin(target_feature, [sample[1] for sample in samples]):  # was not picked already
                #     continue
                if target_feature["label"] != sample["label"]:  # needs to have same label
                    continue
                if np.linalg.norm(sample["std"] - target_feature["std"]) > self.minimum_std_distance:  # is close
                    continue
                target_options.append(target_feature)

            if len(target_options) == 0:
                continue

            random_target_option = random.choice(target_options)
            correspondence.append((sample, random_target_option))

        return correspondence

    def _get_sample_concensus(self, samples):
        bins = 10
        # calculate differences
        samples_differences = np.stack([sample[0]["mean"]-sample[1]["mean"] for sample in samples])

        # get histogram, calculate the max bin
        hist, edges = np.histogramdd(samples_differences,bins)
        if hist.max() < 10:
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

    def visualize_results(self, consensus, target):
        moved = copy.deepcopy(self.down_pcd).transform(self.get_affine_transform(consensus))

        correspondence_linesets = []
        for line in consensus:
            lineset = o3d.geometry.LineSet()
            lineset.points = o3d.utility.Vector3dVector(np.stack([line[0]["mean"], line[1]["mean"]]))
            lineset.lines = o3d.utility.Vector2iVector([[0,1]])
            lineset.paint_uniform_color(np.random.random(3))
            correspondence_linesets.append(lineset)

        correspondence_linesets.extend([moved,target.down_pcd, target.features.pcd, self.down_pcd, self.features.pcd])
        o3d.visualization.draw(correspondence_linesets)

    # def _sample_matches(self, target, samples):
    #     # choose from source a sample that is not in samples randomly
    #     # source_options = [i for i in self.features if np.isin(i, [sample[0] for sample in samples], invert=True)]
    #     random_source_option = random.choice(self.features)
    #
    #     # try to find a sample in target that is not  in samples that matches
    #     target_options = []
    #     for target_feature in target.features:
    #         # if np.isin(target_feature, [sample[1] for sample in samples]):  # was not picked already
    #         #     continue
    #         if target_feature["label"] != random_source_option["label"]:  # needs to have same label
    #             continue
    #         if np.linalg.norm(random_source_option["std"] - target_feature["std"]) > self.minimum_std_distance:  # is close
    #             continue
    #         target_options.append(target_feature)
    #     random_target_option = random.choice(target_options)
    #
    #     return random_source_option, random_target_option

def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    https://gist.github.com/sergeyprokudin/c4bf4059230da8db8256e36524993367
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def test_HyperSkeleton(target_path,source_folder_path,save_path, description=""):
    target = HyperSkeleton(target_path, min_z=0, max_z=1000)
    target.create_global_features()

    subvolume_size = 50
    losses = []
    for file_path in glob.glob(os.path.join(source_folder_path,"*.ply")):
        print(file_path)
        for slicer_index in [0,50,100,150,200]:

            source = HyperSkeleton(file_path,
                                   min_z=slicer_index, max_z=slicer_index+subvolume_size,
                                   cluster_func=target.cluster_func, cluster_colors=target.cluster_colors)
            source.create_global_features()
            # o3d.visualization.draw([target.down_pcd,target.features.pcd,source.down_pcd,source.features.pcd])
            transform, loss = source.register(target)


            losses.append(({"file_name": os.path.basename(file_path),
                            "slice_start": slicer_index,
                            "slice_end": slicer_index+subvolume_size,
                            "loss": loss,
                            "transform":transform}))

            result_dict = {"results":losses, "description":description,
                           "minimum_cluster_std_mean": source.minimum_cluster_std_mean,
                           "minimum_std_distances": source.minimum_std_distance,
                           "minimum_std_distance":source.min_correspondence_percent,
                           "n_clusters":source.n_clusters,
                           "num_bins":source.num_bins,
                           "finish_fit_percent":source.finish_fit_percent,
                           "graph_point_distance":source.graph_point_distance,
                           "max_path_lengths":source.max_path_lengths,
                           "minimum_feature_distance":source.minimum_feature_distance
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
    for i in [0,50,100,150,200]:
        slice_results = [result["loss"] for result in results if result["slice_start"] == i]
        not_outliars = np.array([result for result in slice_results if result > 0 and result < 100])
        print(f"for slice {i}: ")
        print(f"mean results: {not_outliars.mean()} with std of {not_outliars.std()}")
        print( f"Non detection rate is {len([result for result in slice_results if result > 100 or result < 0]) / len(slice_results)}")

    for result in results:
        if result["loss"]==-1:
            print(f"{result['file_name']} didn't match at at slice: {result['slice_start']}")
        elif result["loss"]>100:
            print(f"{result['file_name']} had bad loss of {result['file_name']} at slice: {result['slice_start']}")



if __name__ == "__main__":
    # target = HyperSkeleton(r"D:\datasets\nmdid\clean-body-pcd\case-100114_BONE_TORSO_3_X_3.ply", min_z=0, max_z=1000)
    # target.down_pcd.translate(np.array([500,0,0]))
    # target.create_global_features()
    #
    # source = HyperSkeleton(r"D:\datasets\nmdid\clean-body-pcd\case-121936_BONE_TORSO_3_X_3.ply",
    #                        min_z=50, max_z=150, cluster_func=target.cluster_func,
    #                        cluster_colors=target.cluster_colors)

    # source = HyperSkeleton(r"D:\datasets\nmdid\clean-body-pcd\case-121936_BONE_TORSO_3_X_3.ply",
    #                        min_z=150, max_z=250, cluster_func=target.cluster_func,
    #                        cluster_colors=target.cluster_colors)
    # source.create_global_features()
    # source.register(target)
    #
    # st = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # test_HyperSkeleton(r"D:\datasets\nmdid\clean-body-pcd\case-100114_BONE_TORSO_3_X_3.ply",
    #                    r"D:\datasets\nmdid\clean-body-pcd",
    #                    fr"D:\research_results\HyperSkeleton\{st}_results.json",
    #                    description= "test with full correspondence, now with 16 labels")
    results = r"D:\research_results\HyperSkeleton\2022-01-10_01-07-44_results.json"
    test_view_results(results)