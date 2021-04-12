import networkx as nx
import numpy as np
import copy
import open3d as o3d
import tqdm
import random
from sklearn.cluster import AgglomerativeClustering
import glob, os


class HyperSkeleton:
    def __init__(self, normal_rad=10):
        self.normal_rad = normal_rad
        self.resample = 1
        self.max_iter = 10000
        self.atlas = o3d.geometry.LineSet()
        self.skeleton_graph = nx.DiGraph()
        self.skeleton_graph.add_node(0, prob=0, pos=None, atlas_index=None)
        self.create_atlas()

    def scan(self, pcd_path):
        # get the pcd from path
        pcd = self.load_image(pcd_path)

        # while probability is low
        for _ in range(self.max_iter):

            # get the highest probability node in the graph
            leaf_nodes = [x for x in self.skeleton_graph.nodes() if self.skeleton_graph.out_degree(x) == 0]
            best_prob, best_node = max([(self.skeleton_graph.nodes[leaf_node]["prob"], leaf_node) for
                                        leaf_node in leaf_nodes], key=lambda x: x[0])
            if best_prob < 0.95:
                break

            # expand it
            self._expand_node(best_node, pcd)

        # print results

    def _expand_node(self, node_key, pcd):
        # choose a bone from the atlas
        bone_line, line_index = self._get_bone_from_atlas(node_key)
        bone_line_numpy = bone_line.get_line_coordinate(0)[1] - bone_line.get_line_coordinate(0)[0]
        # filter the non normals
        filtered_pcd = self._remove_non_normals(pcd, bone_line, self.normal_rad)
        # calculate the clusters
        filtered_pcd, clusters_index = self._get_bone_clusters(filtered_pcd, bone_line_numpy)
        # for each cluster
        bone_line_numpy = np.array([bone_line.get_line_coordinate(0)[1] , bone_line.get_line_coordinate(0)[0]])
        for cluster_index in clusters_index:
            # find the best line for the cluster
            points = np.asarray(pcd.points)[cluster_index]
            mean_point = points.mean(axis=0)
            new_bone_line = bone_line_numpy - (bone_line_numpy[1] - bone_line_numpy[0]) / 2 - bone_line_numpy[0] + mean_point  # center point

            # add it to the graph
            lineset = o3d.geometry.LineSet()
            lineset.points = o3d.utility.Vector3dVector(new_bone_line)
            lineset.lines = o3d.utility.Vector2iVector([[0,1]])
            self.skeleton_graph.add_node(str(new_bone_line),pos=lineset, atlas_index=line_index)
            # calculate the probability that this line matches the parent node
            nx.set_node_attributes(self.skeleton_graph,
                                   {str(new_bone_line): {"prob": self._calculate_probability(
                                       str(new_bone_line), node_key)}})
            self.skeleton_graph.add_edge(node_key,str(new_bone_line))
    def _calculate_probability(self, child_node, parent_node):
        return 0
    def _get_bone_from_atlas(self, node_key):
        # take only bones that haven't been selected
        ancestors = nx.ancestors(self.skeleton_graph, node_key)
        preselected_bones_indexes = [self.skeleton_graph.nodes[node]["atlas_index"] for node in ancestors]
        # find bones that are near the preselected, but aren't selected
        preselected_bones = np.asarray(self.atlas.lines)[preselected_bones_indexes]
        options = [option for option in np.asarray(self.atlas.lines) if option[0] in preselected_bones or
                                                                        option[0] in preselected_bones and
                                                                        option not in preselected_bones]
        selected_bone = random.choice(options)


        return o3d.geometry.LineSet(), index


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
        bone_points = np.array([[181.76923077, 313.03846154, 550.23076923],
                                [79.4875, 184.775, 542.45625],
                                [126.25, 163.5, 515.75],
                                [179.00625, 185.44791667, 497.09345238],
                                [239.66666667, 131.33333333, 476.66666667],
                                [237.24444444, 193.60520833, 465.11875],
                                [183.13333333, 275.72982456, 472.12982456]])
        connection = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        self.atlas.points = o3d.utility.Vector3dVector(bone_points)
        self.atlas.lines = o3d.utility.Vector2iVector(connection)
        self.atlas.paint_uniform_color([0, 0, 0])

    def _get_bone_clusters(self, pcd, bone_line, cluster_dist=20):
        bone_tree = o3d.geometry.KDTreeFlann(pcd)
        choosen_points = []
        choosen_indexes = []
        for point_index in tqdm.tqdm(range(np.asarray(pcd.points).shape[0])):
            [k, nie, _] = bone_tree.search_radius_vector_3d(pcd.points[point_index], np.linalg.norm(bone_line) / 2)
            if len(nie) < 5:
                continue

            dists = lineseg_dists(np.asarray(pcd.points)[nie[1:], :],
                                  np.asarray(pcd.points)[point_index] + bone_line / 2,
                                  np.asarray(pcd.points)[point_index] - bone_line / 2)
            if (dists < 1).sum() > np.linalg.norm(bone_line) / 20:
                np.asarray(pcd.colors)[point_index] = [1, 0, 0]
                choosen_points.append(np.asarray(pcd.points)[point_index])
                choosen_indexes.append(point_index)

        choosen_points = np.array(choosen_points)
        choosen_indexes = np.array(choosen_indexes)
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=cluster_dist, linkage="single").fit(
            choosen_points)

        clusters = []
        for cluster_index in range(clustering.n_clusters_):
            cluster = choosen_indexes[clustering.labels_ == cluster_index]
            if cluster.shape[0] < 5:
                continue
            clusters.append(cluster)
        # colors = np.random.random((clustering.labels_.shape[0],3))
        # np.asarray(pcd.colors)[choosen_indexes] = colors[clustering.labels_]

        # print(len(choosen_points))
        return pcd, clusters

    @staticmethod
    def _remove_non_normals(pcd, bone_line, rad=20):
        bone_line = bone_line / np.linalg.norm(bone_line, axis=-1)
        products = np.asarray(pcd.normals).dot(bone_line)
        idx = np.abs(products) < 0.01

        bone_pcd = o3d.geometry.PointCloud()
        bone_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[idx])
        bone_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[idx])
        bone_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[idx])
        return bone_pcd


def lineseg_dists(p, a, b):
    # TODO for you: consider implementing @Eskapp's suggestions
    if np.all(a == b):
        return np.linalg.norm(p - a, axis=1)

    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(p))])
    h = np.expand_dims(h, axis=-1)
    # perpendicular distance component, as before
    # note that for the 3D case these will be vectors
    c = np.cross(p - a, d)

    # use hypot for Pythagoras to improve accuracy
    return np.sqrt(np.sum(np.power(np.hypot(h, c), 2), axis=-1))
