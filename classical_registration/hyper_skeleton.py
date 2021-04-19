import networkx as nx
from networkx.algorithms.dag import dag_longest_path
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
        self.skeleton_graph.add_node(0, prob=0, pos=None, atlas_index=None, loss=0)
        self.pcd = None
        self.create_atlas()
        self.node_name_gen = NodeNameGen()
        self.option_dict = {}  # line_index: [lineset, lineset, lineset...]
        self.loss_stopping_condition = 224
        self.best_result = []
    def scan(self, pcd_path):
        # get the pcd from path
        self.pcd = self.load_image(pcd_path)
        self.calculate_line_options()
        # while probability is low
        for _ in range(self.max_iter):

            # get the highest probability node in the graph
            leaf_nodes = [x for x in self.skeleton_graph.nodes() if self.skeleton_graph.out_degree(x) == 0]
            best_prob, best_node = min([(self.skeleton_graph.nodes[leaf_node]["loss"], leaf_node) for
                                        leaf_node in leaf_nodes], key=lambda x: x[0])

            if self._stopping_conditions(best_node):
                break
            # expand it
            self._expand_node(best_node, self.pcd)

        # print results
        print("finished")
        self.visualize_results()

    def visualize_results(self):
        results = [self.pcd]
        for node in self.best_result:
            if node ==0:
                continue
            line = self.skeleton_graph.nodes[node]["pos"]
            line.paint_uniform_color([1,1,0])
            results.append(self.skeleton_graph.nodes[node]["pos"])

        o3d.visualization.draw_geometries(results)


    def _skeleton_tree_dist(self, u,v,d):
        return self.skeleton_graph.nodes[u]["loss"] + self.skeleton_graph.nodes[u]["loss"]

    def _stopping_conditions(self, best_node):
        best_loss = float("infinity")
        if len(dag_longest_path(self.skeleton_graph)) > len(self.atlas.lines):
            leaf_nodes = [x for x in self.skeleton_graph.nodes() if self.skeleton_graph.out_degree(x) == 0]
            for leaf_node in leaf_nodes:
                loss = self._branch_loss(leaf_node)
                if loss<best_loss:
                    best_loss = loss
        if best_loss <self.loss_stopping_condition:
            self.best_result = nx.shortest_path(self.skeleton_graph, 0, leaf_node)
            return True
        return False
    def _branch_loss(self, end_node):
        path = nx.shortest_path(self.skeleton_graph, 0, end_node)
        if len(path) > len(self.atlas.lines):
            loss = 0
            for node in path:
                loss += self.skeleton_graph.nodes[node]["loss"]
            return loss
        else:
            return float("infinity")
    def calculate_line_options(self):
        for line_index in range(np.asarray(self.atlas.lines).shape[0]):
            line_options = []
            bone_line = np.asarray(self.atlas.lines)[line_index]
            bone_line_numpy = self.atlas.get_line_coordinate(bone_line[0])[1] -\
                              self.atlas.get_line_coordinate(bone_line[0])[0]
            # filter the non normals
            filtered_pcd = self._remove_non_normals(self.pcd, bone_line_numpy, self.normal_rad)
            # calculate the clusters
            filtered_pcd, clusters_index = self._get_bone_clusters(filtered_pcd, bone_line_numpy)
            # for each cluster
            bone_line_numpy = np.array([self.atlas.get_line_coordinate(bone_line[0])[1], self.atlas.get_line_coordinate(bone_line[0])[0]])
            for cluster_index in clusters_index:

                # find the best line for the cluster
                points = np.asarray(filtered_pcd.points)[cluster_index]
                mean_point = points.mean(axis=0)
                new_bone_line = bone_line_numpy - (bone_line_numpy[1] + bone_line_numpy[0]) / 2 + mean_point  # center point

                # add it to the list
                lineset = o3d.geometry.LineSet()
                lineset.points = o3d.utility.Vector3dVector(new_bone_line)
                lineset.lines = o3d.utility.Vector2iVector([[0, 1]])

                line_options.append(lineset)
            self.option_dict[line_index] = line_options

    def _expand_node(self, node_key, pcd):
        # if we been on all bones, don't expand
        if len(nx.ancestors(self.skeleton_graph, node_key)) == len(self.atlas.lines):
            return
        _, line_index = self._get_bone_from_atlas(node_key)
        # choose a bone from the atlas

        for line_option in self.option_dict[line_index]:
            node_name = self.node_name_gen()
            self.skeleton_graph.add_node(node_name, pos=line_option, atlas_index=line_index)
            # calculate the probability that this line matches the parent node
            self.skeleton_graph.add_edge(node_key, node_name)
            nx.set_node_attributes(self.skeleton_graph,
                                   {node_name: {"loss": self._calculate_loss(node_name)}})


    def _calculate_loss(self, child_node):
        # get current bone indexes
        current_bone_index = np.asarray(self.atlas.lines)[self.skeleton_graph.nodes[child_node]["atlas_index"]]
        loss = []
        # get the rest of the bones from the tree, that connect to the current bone
        ancestors = nx.ancestors(self.skeleton_graph, child_node)
        if ancestors == {0}:
            return 0  # if no ancestors, loss is zero

        # find where the ancestors connect (if they do)
        for node in ancestors:
            if node == 0:
                continue

            line_index = np.asarray(self.atlas.lines)[self.skeleton_graph.nodes[node]["atlas_index"]]
            if line_index[0] in current_bone_index:
                match_index = (0, np.where(line_index[0] == current_bone_index)[0][0])
            elif line_index[1] in current_bone_index:
                match_index = (1, np.where(line_index[1] == current_bone_index)[0][0])
            else:
                continue

            # calculate the loss
            parent_lineset = self.skeleton_graph.nodes[node]["pos"]
            parent_pos = parent_lineset.get_line_coordinate(0)[match_index[0]]
            child_lineset = self.skeleton_graph.nodes[child_node]["pos"]
            child_pos = child_lineset.get_line_coordinate(0)[match_index[1]]
            loss.append(np.linalg.norm(parent_pos-child_pos))

        if len(loss) == 0:
            total_loss = 5
        else:
            total_loss = sum(loss)/len(loss)
        return total_loss

    def _get_bone_from_atlas(self, node_key):
        # take only bones that haven't been selected
        ancestors = nx.ancestors(self.skeleton_graph, node_key)
        ancestors.add(node_key)
        ancestors.remove(0)
        if len(ancestors) > 0:
            preselected_bones_indexes = [self.skeleton_graph.nodes[node]["atlas_index"] for node in ancestors]
            # find bones that are near the preselected, but aren't selected
            preselected_bones = np.asarray(self.atlas.lines)[preselected_bones_indexes]
            options = [option for option in np.asarray(self.atlas.lines) if (option[0] in preselected_bones or
                       option[1] in preselected_bones) and
                       option not in preselected_bones]
        else:
            options = list(np.asarray(self.atlas.lines))
        if len(options) == 0:
            print ("error")
        selected_bone = random.choice(options)

        index = np.where(np.asarray(self.atlas.lines) == selected_bone)[0][0]
        new_bone = o3d.geometry.LineSet()
        new_bone.points = o3d.utility.Vector3dVector(np.asarray(self.atlas.points)[selected_bone])
        new_bone.lines = o3d.utility.Vector2iVector([[0, 1]])
        return new_bone, index

    def load_image(self, path):
        pcd = o3d.io.read_point_cloud(path)

        numpy_source = np.asarray(pcd.points)
        numpy_source = numpy_source[numpy_source[:, 2] > 300]
        numpy_source = numpy_source[numpy_source[:, 2] < 601]

        pcd_index = np.random.randint(0, numpy_source.shape[0], int(numpy_source.shape[0] * self.resample))
        pcd.points = o3d.utility.Vector3dVector(numpy_source[pcd_index])
        pcd.paint_uniform_color([0.1, 0.1, 0.1])
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_rad, max_nn=30))

        return pcd

    def create_atlas(self):
        # resulting
        # lines
        # [[187. 282. 456.]
        #  [231. 187. 466.]]
        # [[236. 177. 468.]
        #  [204. 166. 480.]]
        # [[219. 165. 475.]
        #  [186. 180. 480.]]
        # [[168. 189. 485.]
        #  [117. 181. 507.]]
        # [[119. 183. 504.]
        #  [126. 236. 501.]]
        # [[161. 300. 558.]
        #  [74. 194. 548.]]
        # [[124. 244. 501.]
        #  [149. 272. 491.]]
        # [[69. 220. 448.]
        #  [66. 133. 410.]]
        # [[347. 264. 452.]
        #  [273. 172. 466.]]
        bone_points = np.array([[181.76923077, 313.03846154,550.23076923],
                                [79.4875, 184.775, 542.45625],
                                [126.25, 163.5, 515.75],
                                [179.00625, 185.44791667, 497.09345238],
                                [239.66666667, 131.33333333, 476.66666667],
                                [237.24444444, 193.60520833, 465.11875],
                                [183.13333333, 275.72982456, 472.12982456]])
        connection = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],[5,6]])
        self.atlas.points = o3d.utility.Vector3dVector(bone_points)
        self.atlas.lines = o3d.utility.Vector2iVector(connection)
        self.atlas.paint_uniform_color([0, 1, 1])

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
            if cluster.shape[0] < 10:
                continue
            clusters.append(cluster)
        colors = np.random.random((clustering.labels_.shape[0],3))
        np.asarray(pcd.colors)[choosen_indexes] = colors[clustering.labels_]

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
if __name__ == "__main__":
    hs = HyperSkeleton()
    hs.scan(r"D:\visceral\full_skeletons\102946_CT_Wb.ply")