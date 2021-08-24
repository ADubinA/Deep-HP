import json

import networkx as nx
from networkx.algorithms.dag import dag_longest_path
import numpy as np
import copy
import open3d as o3d
import tqdm
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
import glob, os
from create_atlas import histogram_features, create_rfb

class HyperSkeleton:
    def __init__(self, normal_rad=20, slice_size = 1000):
        self.normal_rad = normal_rad
        self.resample = 0.2
        self.stop_after_n_results =300
        self.max_iter = 10000
        self.space_loss_coef = 100
        self.atlas_axises = []
        self.atlas = {}
        self.atlas_path = r"D:\experiments\atlases\atlas_1.json"
        self.rbfs = None
        self.skeleton_graph = nx.DiGraph()
        self.skeleton_graph.add_node(0, prob=0, pos=None, atlas_index=None, loss=0,end_node=False)
        self.pcd = None
        self.histogram_pcd= None
        self.option_dict = {}  # {axis_index:["mean":numpy(3), "std":numpy(3), ]} axis_index should be match atlas when not missing
        self.best_result = []  # list of nodes on the tree
        self.all_options=[] # list of tuple that has all the options in self.option_dict. of the form [(axis_index,option_index)]
        self.node_name_gen = NodeNameGen()
        self.slice_size =slice_size
        self.slice_start =0# np.random.random()*(numpy_source[:,2].max()-self.slice_size)
        self.create_atlas()

    def nearest_scan(self, pcd_path,histogram_path = None):
        self.load_image(pcd_path,histogram_path=histogram_path)
        self.calculate_line_options()
        std_thresh=3

        # for each option calculated
        matches = []
        for axis_index, source_options in self.option_dict.items():
            for source_option in source_options:

                # check each distribution on the atlas to see where is best fit
                for atlas_option in self.atlas[axis_index]:
                    if np.linalg.norm(atlas_option["std"]-source_option["std"]) > std_thresh:
                        continue

                    # calculate loss if we moved the source distribution to atlas distribution
                    matches.append((atlas_option, source_option))
        transforms = np.array([match[0]["mean"]-match[1]["mean"] for match in matches])
        H, edges = np.histogramdd(transforms, bins=10)
        if H.max()<=2:
            return []
        best_options_index = np.transpose(np.nonzero((H>H.max()/2)))

        # grid_transforms = [np.array([edges[i][best_option_index[i]] for i in range(3)])
        #                   for best_option_index in best_options_index]
        matches_list = []
        for best_option_index in best_options_index:
            start = np.array([edges[i][best_option_index[i]] for i in range(3)])
            end = np.array([edges[i][best_option_index[i]+1] for i in range(3)])
            matched_indexes = np.where(np.logical_and(np.all(transforms >= start,axis=-1),np.all(transforms < end,axis=-1)))[0]
            matches_list.append([matches[i] for i in matched_indexes.tolist()])
        return matches_list
        # find outliers

    def tree_scan(self, pcd_path,histogram_path = None):
        # get the pcd from path
        self.load_image(pcd_path,histogram_path=histogram_path)
        self.calculate_line_options()
        # while probability is low
        print("starting to match")
        for _ in tqdm.tqdm(range(self.max_iter)):

            # get the highest probability node in the graph
            leaf_nodes = [x for x in self.skeleton_graph.nodes() if self.skeleton_graph.out_degree(x) == 0 and
                          not self.skeleton_graph.nodes[x]["end_node"]]
            best_prob, best_node = min([(self.skeleton_graph.nodes[leaf_node]["loss"], leaf_node) for
                                        leaf_node in leaf_nodes], key=lambda x: x[0])

            self._calculate_best_option()

            if self._stopping_conditions():
                break
            # expand it
            self._expand_node(best_node)

        # print results
        print("finished")

    def visualize_graph(self):
        pos = hierarchy_pos(self.skeleton_graph, 0)
        nx.draw(self.skeleton_graph, pos,labels=nx.get_node_attributes(self.skeleton_graph,"text"))

    def transform(self, transform, is_translate=True):
        if is_translate:
            self.pcd.translate(transform)
            for node in self.best_result:
                self.skeleton_graph.nodes[node]["pos"].translate(transform)

    def vis_all_results(self):
        leaf_nodes = [x for x in self.skeleton_graph.nodes() if self.skeleton_graph.out_degree(x) == 0 and
                                                                self.skeleton_graph.nodes[x]["end_node"]]
        leaf_nodes = sorted(leaf_nodes,key=lambda x:self.skeleton_graph.nodes[x]["loss"])
        for leaf_node in leaf_nodes:
            results = [self.pcd]
            for node in nx.shortest_path(self.skeleton_graph,0,leaf_node):
                if node ==0:
                    continue
                line = self.skeleton_graph.nodes[node]["pos"]
                line.paint_uniform_color([1,1,0])
                results.append(line)

            o3d.visualization.draw_geometries(results)

    def visualize_atlas(self, return_list=False):
        gauss_list = []
        for line, line_group in self.atlas.items():
            gauss_list.extend(line_group)
        return self._visualize_gaussians(gauss_list, return_list)

    def _visualize_gaussians(self, gauss_list, return_list=False):
        meshes = []
        for rbf in gauss_list:
            meshes.append(create_rfb(rbf["mean"],rbf["std"],rbf["axis"],rbf["color"] ))
        if return_list:
            return meshes
        else:
            meshes.append(self.pcd)
            o3d.visualization.draw(meshes)
            return None

    def visualize_best_results(self, save_folder=None, return_list=False):

        results = [self.pcd]
        if save_folder is not None:
            # o3d.io.write_line_set(os.path.join(save_folder,"atlas.ply"),self.atlas)
            o3d.io.write_point_cloud(os.path.join(save_folder,"pcd.ply"),self.pcd)

        gauss_list = []
        for node in self.best_result:
            if node ==0:
                continue
            node_data = self.skeleton_graph.nodes[node]
            gauss_list.append({"mean": node_data["mean"],
                               "std": node_data["std"],
                               "axis": self.atlas_axises[node_data["option_index"][0]],
                               "color": self.atlas[node_data["option_index"][0]][node_data["option_index"][1]]["color"]})
            results.extend(self._visualize_gaussians(gauss_list,True))
            if save_folder is not None:
                o3d.io.write_line_set(os.path.join(save_folder,str(node)+".ply"), )
        results.extend(self.visualize_atlas(True))
        if return_list:
            return results
        else:
            o3d.visualization.draw_geometries(results)

    def _skeleton_tree_dist(self, u,v,d):
        return self.skeleton_graph.nodes[u]["loss"] + self.skeleton_graph.nodes[u]["loss"]

    def _calculate_best_option(self):
        best_loss = float("infinity")
        leaf_nodes = [x for x in self.skeleton_graph.nodes() if self.skeleton_graph.nodes[x]["end_node"]]
        if len(leaf_nodes)>0:
            best_leaf = None
            for leaf_node in leaf_nodes:
                loss = self.skeleton_graph.nodes[leaf_node]["loss"]
                if loss<best_loss:
                    best_loss = loss
                    best_leaf = leaf_node
            self.best_result = nx.shortest_path(self.skeleton_graph, 0, best_leaf)

    def _stopping_conditions(self):
        # get all leaf nodes that are not end nodes
        if len(self.best_result) < self._num_of_matches()-1:
            return False
        undiscovered_nodes = [self.skeleton_graph.nodes[x]["loss"] for x in self.skeleton_graph.nodes()
                              if self.skeleton_graph.out_degree(x) == 0 and not self.skeleton_graph.nodes[x]["end_node"]]
        best_end_node_loss = self.skeleton_graph.nodes[self.best_result[-1]]["loss"]


        if best_end_node_loss <= min(undiscovered_nodes):
            return True
        end_nodes = [x for x in self.skeleton_graph.nodes() if self.skeleton_graph.nodes[x]["end_node"]]
        if len(end_nodes) > self.stop_after_n_results:
            return True
        return False

    def _num_of_matches(self):
        num_matches =0
        for axis, option in self.option_dict.items():
            num_matches +=len(option)
        return num_matches

    def calculate_line_options(self):
        if not self.histogram_pcd:
            self.histogram_pcd = histogram_features(self.pcd, self.normal_rad/2,0)[1]

        for axis_index in range(len(self.atlas_axises)):
            axis = self.atlas_axises[axis_index]
            line_options = []
            # calculate the clusters
            filtered_pcd, clusters_index = self._get_bone_clusters( self.histogram_pcd.__copy__(), axis)
            if filtered_pcd is None:
                print(f"axis: {axis} was not found")
                continue

            # for each cluster
            i = 0
            for cluster_index in clusters_index:

                # find the best line for the cluster
                points = np.asarray(filtered_pcd.points)[cluster_index]
                line_options.append({"mean": points.mean(axis=0), "std": points.std(axis=0),
                                     "axis": axis, "color": np.array([1,0,0])})
                self.all_options.append((axis_index,i))
                i += 1
            self.option_dict[axis_index] = line_options

    def _expand_node(self, node_key):
        # if we been on all bones, don't expand
        if len(nx.ancestors(self.skeleton_graph, node_key)) == self._num_of_matches()-1:
            nx.set_node_attributes(self.skeleton_graph, {node_key: {"end_node":True}})
            return
        options_tuple = self._get_bone_from_atlas(node_key)
        # choose a bone from the atlas

        for option_tuple in options_tuple:
            node_name = self.node_name_gen()
            self.skeleton_graph.add_node(node_name,
                                         option_index=option_tuple, #atlas_index>>>>>>
                                         mean=self.option_dict[option_tuple[0]][option_tuple[1]]["mean"],
                                         std=self.option_dict[option_tuple[0]][option_tuple[1]]["std"],
                                         end_node=False
                                         )
            # calculate the probability that this line matches the parent node
            self.skeleton_graph.add_edge(node_key, node_name)
            loss,best_atlas_index, best_transform,feature_loss = self._loss(node_name)
            nx.set_node_attributes(self.skeleton_graph,
                                   {node_name: {"loss": loss, "text": int(loss),"atlas_index":best_atlas_index,
                                                "transform":best_transform,"feature_loss":feature_loss}})

    def _loss(self, child_node):
        # get current bone indexes
        loss = []
        # get the rest of the bones from the tree, that connect to the current bone
        ancestors = nx.ancestors(self.skeleton_graph, child_node)
        ancestors.remove(0)
        axis_index = self.skeleton_graph.nodes[child_node]["option_index"][0]
        atlas_options = self.atlas[axis_index]

        # for calculating the best loss
        losses = []
        for atlas_option in atlas_options:
            # calculate the loss of the current
            feature_loss= np.linalg.norm(self.skeleton_graph.nodes[child_node]["std"] - atlas_option["std"])
            child_transform = self.skeleton_graph.nodes[child_node]["mean"] - atlas_option["mean"]

            # if this is the first node, no loss is added from transform
            # else
            transform_list = [child_transform]
            feature_loss_list = [feature_loss]
            for node in ancestors:
                transform_list.append(self.skeleton_graph.nodes[node]["transform"])
                feature_loss_list.append(self.skeleton_graph.nodes[node]["feature_loss"])
            option_transform = np.array(transform_list).mean(axis=0)

            transform_list = [option_transform - child_transform]
            for node in ancestors:
                transform_list.append(option_transform - self.skeleton_graph.nodes[node]["transform"])
            space_loss = np.linalg.norm(np.array(transform_list).mean(axis=0))
            all_feature_loss = np.linalg.norm(np.array(feature_loss_list).mean(axis=0))

            losses.append({"total_loss": space_loss * self.space_loss_coef + all_feature_loss,
                           "feature_loss":feature_loss,
                           "transform": option_transform})
        best_option = losses.index(min(losses, key=lambda x: x["total_loss"]))

        return losses[best_option]["total_loss"],(axis_index,best_option),\
               losses[best_option]["transform"], losses[best_option]["feature_loss"]

    def _get_bone_from_atlas(self, node_key):
        # take only bones that haven't been selected
        ancestors = nx.ancestors(self.skeleton_graph, node_key)
        ancestors.add(node_key)
        ancestors.remove(0)
        # if len(ancestors) > 0:
        preselected_bones_indexes = [self.skeleton_graph.nodes[node]["option_index"] for node in ancestors]
        # find bones that are in the same line, but aren't selected
        options = [option for option in self.all_options if option not in preselected_bones_indexes]
        return options

    def load_image(self, path, histogram_path=None):
        pcd = o3d.io.read_point_cloud(path)

        numpy_source = np.asarray(pcd.points)
        numpy_source = numpy_source[numpy_source[:, 2] > self.slice_start]
        numpy_source = numpy_source[numpy_source[:, 2] < self.slice_start+self.slice_size]

        pcd_index = np.random.randint(0, numpy_source.shape[0], int(numpy_source.shape[0] * self.resample))
        pcd.points = o3d.utility.Vector3dVector(numpy_source[pcd_index])
        pcd.paint_uniform_color([0.1, 0.1, 0.1])
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_rad, max_nn=30))

        if histogram_path:
            self.histogram_pcd = o3d.io.read_point_cloud(histogram_path)
        self.pcd = pcd

    def create_atlas(self):
        with open(self.atlas_path, 'r') as f:
            atlas_file = json.load(f)["atlas"]
        self.atlas_axises = []
        self.atlas = {}
        i = 0
        for line, line_group in atlas_file.items():
            self.atlas_axises.append(np.array(eval(line)))
            rbfs = []
            for rbf in line_group:
                rbfs.append({
                    "mean": np.array(rbf["mean"]),
                    "std": np.array(rbf["std"]),
                    "axis": np.array(rbf["axis"]),
                    "index": np.array(rbf["index"]),
                    "color": np.array(rbf["color"])})
            self.atlas[i] = rbfs
            i+=1

    def _get_bone_clusters(self, pcd, bone_line, cluster_dist=5):

        bone_tree = o3d.geometry.KDTreeFlann(pcd)
        pcd.paint_uniform_color([0,0,0])
        choosen_points = []
        choosen_indexes = []
        for point_index in tqdm.tqdm(range(np.asarray(pcd.points).shape[0])):
            [k, nie, _] = bone_tree.search_radius_vector_3d(pcd.points[point_index], np.linalg.norm(bone_line) / 2)
            if len(nie) < 5:
                continue

            dists = lineseg_dists(np.asarray(pcd.points)[nie[1:], :],
                                  np.asarray(pcd.points)[point_index] + bone_line / 2,
                                  np.asarray(pcd.points)[point_index] - bone_line / 2)
            if (dists < 3).sum() > np.linalg.norm(bone_line)/2:
                np.asarray(pcd.colors)[point_index] = [1, 0, 0]
                choosen_points.append(np.asarray(pcd.points)[point_index])
                choosen_indexes.append(point_index)

        if len(choosen_points)<3:
            return None,None

        choosen_points = np.array(choosen_points)
        choosen_indexes = np.array(choosen_indexes)
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=cluster_dist, linkage="single").fit(
            choosen_points)

        clusters = []
        for cluster_index in range(clustering.n_clusters_):
            cluster = choosen_indexes[clustering.labels_ == cluster_index]
            if cluster.shape[0] < 5:
                np.asarray(pcd.colors)[cluster] = np.zeros(3)
                continue
            if np.asarray(pcd.points)[cluster].std(axis=0).max() < 10:
                np.asarray(pcd.colors)[cluster] = np.zeros(3)
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


def lineseg_dists(p, a, b, clamp=True):

    # TODO for you: consider implementing @Eskapp's suggestions
    if clamp:
        if np.all(a == b):
            return np.linalg.norm(p - a, axis=-1)

        # normalized tangent vector
        d = np.divide(b - a, np.linalg.norm(b - a))

        # signed parallel distance components
        s = np.dot(a - p, d)
        t = np.dot(p - b, d)

        # clamped parallel distance
        h = np.maximum.reduce([s, t, np.zeros(len(p))])

        h = np.expand_dims(h,axis = -1)
        # perpendicular distance component, as before
        # note that for the 3D case these will be vectors
        c = np.cross(p - a, d)

        # use hypot for Pythagoras to improve accuracy
        return np.sqrt(np.sum(np.power(np.hypot(h, c),2), axis=-1))
    else:
        return np.linalg.norm(np.cross(b-a, a-p), axis=-1)/np.linalg.norm(b-a, axis=-1)


class NodeNameGen:
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 1
        return self.value


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def calculate_deformation(source, target):
    """

    :param source: what to transform
    :type source: HyperSkeleton
    :param target: where to transform
    :type target: HyperSkeleton
    :return:
    :rtype:
    """
    # check that both have a best result
    if len(source.best_result) == 0 or len(target.best_result) == 0:
        raise ValueError("Skeletons faild to find results, or weren't processes.")
    if 0 in source.best_result: source.best_result.remove(0)
    if 0 in target.best_result: target.best_result.remove(0)
    # get the skeletons that were found both in each skeleton
    nodes_index_in_source = set(source.skeleton_graph.nodes[node]["atlas_index"] for node in source.best_result)
    nodes_index_in_target = set(target.skeleton_graph.nodes[node]["atlas_index"] for node in target.best_result)
    match_points_line_index = list(nodes_index_in_target.intersection(nodes_index_in_source))

    # find outliers
    # average transform
    transforms = []
    for line_index in match_points_line_index:
        source_node = [node for node in source.best_result
                        if source.skeleton_graph.nodes[node]["atlas_index"] == line_index][0]
        target_node = [node for node in target.best_result
                          if target.skeleton_graph.nodes[node]["atlas_index"] == line_index][0]
        source_lineset = source.skeleton_graph.nodes[source_node]["pos"]
        source_location =  np.asarray(source_lineset.points)[1]-np.asarray(source_lineset.points)[0]/2
        target_lineset = target.skeleton_graph.nodes[target_node]["pos"]
        target_location =  np.asarray(target_lineset.points)[1]-np.asarray(target_lineset.points)[0]/2

        transforms.append(target_location - source_location)
    transform = np.array(transforms).mean(axis=0)
    return transform


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


def generate_data(data_path, atlas_path):
    save_path = r"D:\experiments\data_gen_skeletons\labels_1.json"
    summery = "results of aug 21 atlas"
    num_samples = 100
    times_per_sample = 1
    log_rate=1
    atlas = o3d.io.read_point_cloud(atlas_path)
    log = []
    for path in glob.glob(os.path.join(data_path,"*.ply")):
        for i in range(times_per_sample):
            hs = HyperSkeleton()
            matches = hs.nearest_scan(path, histogram_path=None)
            transforms = [np.array([points[0]["mean"]-points[1]["mean"] for points in match]).mean(axis=0) for match in matches]
            best_loss = (float("infinity"),0)
            for transform_index in range(len(transforms)):
                source = hs.pcd.__copy__()
                source.translate(transforms[transform_index])
                loss = chamfer_distance(np.asarray(source.points),np.array(atlas.points),direction="x_to_y")
                if loss < best_loss[0]:
                    best_loss = (loss, transform_index)
            if len(transforms) !=0:
                log.append({"path": os.path.basename(path), "start":hs.slice_start,"loss": best_loss[0],
                            "affine_transform": transforms[best_loss[1]],
                            "matches":matches[best_loss[1]]})
            else:
                log.append({"path": os.path.basename(path), "start":hs.slice_start,"loss":float("infinity"),
                            "affine_transform": None})

            if len(log) % log_rate == 0:
                results = {"results": log, "slice_size": hs.slice_size}
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4,cls=NumpyEncoder)

        if len(log)>num_samples:
            break
    results = {"results":log, "slice_size":hs.slice_size, "summery": summery}
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4,cls=NumpyEncoder)


def visualize_results(json_path):
    import matplotlib.pyplot as plt
    with open(json_path) as f:
       data_dict = json.load(f)

    # how many where not detected
    x_good= []
    y_good=[]
    x_bad = []
    y_bad = []
    for result in data_dict["results"]:
        if result["transform"] == None:
            x_bad.append(result["start"])
            y_bad.append(0)
        else:
            x_good.append((result["start"]))
            y_good.append((result["loss"]))

    plt.plot(x_bad, y_bad, 'bo')
    plt.plot(x_good, y_good, 'go')
    plt.show()


    # historgram of accuracy per slice range


class NumpyEncoder(json.JSONEncoder):
    """https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
if __name__ == "__main__":
    # atlas = o3d.io.read_point_cloud(r"D:\visceral\full_skeletons\102946_CT_Wb.ply")
    # hs1 = HyperSkeleton()
    # # exp_save = r"D:\experiments\reg1"
    # matches = hs1.nearest_scan(r"D:\visceral\full_skeletons\102850_CT_Wb.ply", histogram_path = None)#r'D:\visceral\testing\102946_CT_Wb_histogram.ply')
    # for match in matches:
    #     source = hs1.pcd.__copy__()
    #     source.translate(np.array([points[0]["mean"]-points[1]["mean"] for points in match]).mean(axis=0))
    #     o3d.visualization.draw_geometries([ source,atlas])
    # hs1.visualize_atlas()
    # hs1.visualize_best_results()
    generate_data(r"D:\visceral\full_skeletons",r"D:\visceral\full_skeletons\102946_CT_Wb.ply")
    # visualize_results(r"D:\experiments\data_gen_skeletons\test1.json")