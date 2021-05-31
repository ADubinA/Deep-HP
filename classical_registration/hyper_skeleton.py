import networkx as nx
from networkx.algorithms.dag import dag_longest_path
import numpy as np
import copy
import open3d as o3d
import tqdm
import random
from sklearn.cluster import AgglomerativeClustering
import glob, os
from create_atlas import histogram_features

class HyperSkeleton:
    def __init__(self, normal_rad=10):
        self.normal_rad = normal_rad
        self.resample = 0.1
        self.stop_after_n_results =30
        self.max_iter = 10000
        self.loss_mse_value = 0.1
        self.atlas = o3d.geometry.LineSet()
        self.rbfs = None
        self.skeleton_graph = nx.DiGraph()
        self.skeleton_graph.add_node(0, prob=0, pos=None, atlas_index=None, loss=0,end_node=False)
        self.pcd = None
        self.create_atlas()
        self.node_name_gen = NodeNameGen()
        self.option_dict = {}  # line_index: [{"pos":lineset, "transform":numpy(3), "mean":numpy(3), "std":numpy(3)},]
        self.best_result = []

    def scan(self, pcd_path):
        # get the pcd from path
        self.pcd = self.load_image(pcd_path)
        self.calculate_line_options()
        # while probability is low
        print("starting to match")
        for _ in tqdm.tqdm(range(self.max_iter)):

            # get the highest probability node in the graph
            leaf_nodes = [x for x in self.skeleton_graph.nodes() if self.skeleton_graph.out_degree(x) == 0 and
                          not self.skeleton_graph.nodes[x]["end_node"]]
            best_prob, best_node = min([(self.skeleton_graph.nodes[leaf_node]["loss"], leaf_node) for
                                        leaf_node in leaf_nodes], key=lambda x: x[0])

            if self._stopping_conditions(best_node):
                break
            # expand it
            self._expand_node(best_node, self.pcd)

        # print results
        print("finished")

    def visualize_graph(self):
        pos = hierarchy_pos(self.skeleton_graph, 0)
        nx.draw(self.skeleton_graph, pos,labels=nx.get_node_attributes(self.skeleton_graph,"text"))

    def vis_all_results(self):
        leaf_nodes = [x for x in self.skeleton_graph.nodes() if self.skeleton_graph.out_degree(x) == 0 and
                                                                self.skeleton_graph.nodes[x]["end_node"]]
        leaf_nodes = sorted(leaf_nodes,key=lambda x:self.skeleton_graph.nodes[x]["loss"])
        for leaf_node in leaf_nodes:
            results = [self.pcd,self.atlas]
            for node in nx.shortest_path(self.skeleton_graph,0,leaf_node):
                if node ==0:
                    continue
                line = self.skeleton_graph.nodes[node]["pos"]
                line.paint_uniform_color([1,1,0])
                results.append(line)

            o3d.visualization.draw_geometries(results)

    def visualize_results(self):
        save_folder = "D:\experiments\skeleton_reg_1"
        results = [self.pcd,self.atlas]
        o3d.io.write_line_set(os.path.join(save_folder,"atlas.ply"),self.atlas)
        o3d.io.write_point_cloud(os.path.join(save_folder,"pcd.ply"),self.pcd)

        for node in self.best_result:
            if node ==0:
                continue
            line = self.skeleton_graph.nodes[node]["pos"]
            line.paint_uniform_color([1,1,0])
            results.append(line)
            o3d.io.write_line_set(os.path.join(save_folder,str(node)+".ply"), line)

        o3d.visualization.draw_geometries(results)
        self.visualize_graph()

    def _skeleton_tree_dist(self, u,v,d):
        return self.skeleton_graph.nodes[u]["loss"] + self.skeleton_graph.nodes[u]["loss"]

    def _stopping_conditions(self, best_node):
        best_loss = float("infinity")
        if len(dag_longest_path(self.skeleton_graph)) > self._num_of_matches():
            leaf_nodes = [x for x in self.skeleton_graph.nodes() if self.skeleton_graph.nodes[x]["end_node"]]
            best_leaf = None
            for leaf_node in leaf_nodes:
                loss = self._branch_loss(leaf_node)
                if loss<best_loss:
                    best_loss = loss
                    best_leaf = leaf_node
                    self.best_result = nx.shortest_path(self.skeleton_graph, 0, best_leaf)

            if len(leaf_nodes)>self.stop_after_n_results:
                return True
        return False
    def _branch_loss(self, end_node):
        path = nx.shortest_path(self.skeleton_graph, 0, end_node)
        if len(path) > self._num_of_matches():
            loss = 0
            for node in path:
                loss += self.skeleton_graph.nodes[node]["loss"]
            return loss
        else:
            return float("infinity")
    def _num_of_matches(self):
        return len(self.option_dict.keys()) - 1
    def calculate_line_options(self):
        histogram_pcd = histogram_features(self.pcd, self.normal_rad,0)[1]
        for line_index in range(np.asarray(self.atlas.lines).shape[0]):
            line_options = []
            transforms = []
            bone_line_numpy = self.atlas.get_line_coordinate(line_index)[1] - self.atlas.get_line_coordinate(line_index)[0]
            # filter the non normals
            # filtered_pcd = self._remove_non_normals(self.pcd, bone_line_numpy, self.normal_rad)
            # calculate the clusters
            filtered_pcd, clusters_index = self._get_bone_clusters(histogram_pcd.__copy__(), bone_line_numpy)
            if filtered_pcd is None:
                print(f"bone with index: {line_index} was not found")
                continue
            # for each cluster
            bone_line_numpy = np.array([self.atlas.get_line_coordinate(line_index)[1], self.atlas.get_line_coordinate(line_index)[0]])
            for cluster_index in clusters_index:

                # find the best line for the cluster
                points = np.asarray(filtered_pcd.points)[cluster_index]
                mean_point = points.mean(axis=0)
                transform = (bone_line_numpy[1] + bone_line_numpy[0]) / 2 - mean_point
                new_bone_line = bone_line_numpy - transform  # transform to the mean of the cluster, center of line

                # add it to the list
                lineset = o3d.geometry.LineSet()
                lineset.points = o3d.utility.Vector3dVector(new_bone_line)
                lineset.lines = o3d.utility.Vector2iVector([[0, 1]])

                line_options.append({"pos":lineset, "transform":transform, "mean":mean_point, "std":points.std(axis=0)})
            self.option_dict[line_index] = line_options


    def _expand_node(self, node_key, pcd):
        # if we been on all bones, don't expand
        if len(nx.ancestors(self.skeleton_graph, node_key)) == self._num_of_matches():
            nx.set_node_attributes(self.skeleton_graph,
                                   {node_key: {"end_node":True}})
            return
        _, line_index = self._get_bone_from_atlas(node_key)
        # choose a bone from the atlas

        for line_option_index in range(len(self.option_dict[line_index])):
            node_name = self.node_name_gen()
            self.skeleton_graph.add_node(node_name, pos=self.option_dict[line_index][line_option_index]["pos"],
                                        atlas_index=line_index,
                                         transform=self.option_dict[line_index][line_option_index]["transform"],
                                         mean=self.option_dict[line_index][line_option_index]["mean"],
                                         std=self.option_dict[line_index][line_option_index]["std"],
                                         end_node=False
                                         )
            # calculate the probability that this line matches the parent node
            self.skeleton_graph.add_edge(node_key, node_name)
            loss = self._loss_mse(node_name)
            nx.set_node_attributes(self.skeleton_graph,
                                   {node_name: {"loss": loss, "text": int(loss)}})


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

    def _loss_mse(self, child_node):
        # get current bone indexes
        node_transform = self.skeleton_graph.nodes[child_node]["transform"]
        loss = []
        # get the rest of the bones from the tree, that connect to the current bone
        ancestors = nx.ancestors(self.skeleton_graph, child_node)
        ancestors.remove(0)

        for node in ancestors:
            node_line_index = self.skeleton_graph.nodes[node]["atlas_index"]
            result_lineset = self.skeleton_graph.nodes[node]["pos"]
            atlas_lineset = self.atlas.get_line_coordinate(node_line_index)
            space_loss = np.linalg.norm((atlas_lineset - node_transform) - result_lineset.get_line_coordinate(0))
            rbf_loss = np.linalg.norm(self.skeleton_graph.nodes[node]["std"] - self.rbfs[node_line_index]["std"])
            loss.append(rbf_loss+space_loss*self.loss_mse_value)

        if len(loss) == 0:
            total_loss = 0
        else:
            total_loss = sum(loss)/len(loss)

        node_line_index = self.skeleton_graph.nodes[child_node]["atlas_index"]
        self_loss = rbf_loss = np.linalg.norm(
            self.skeleton_graph.nodes[child_node]["std"] - self.rbfs[node_line_index]["std"])
        total_loss +=self_loss
        return total_loss

    def _get_bone_from_atlas(self, node_key):
        # take only bones that haven't been selected
        ancestors = nx.ancestors(self.skeleton_graph, node_key)
        ancestors.add(node_key)
        ancestors.remove(0)
        # if len(ancestors) > 0:
        preselected_bones_indexes = [self.skeleton_graph.nodes[node]["atlas_index"] for node in ancestors]
        # find bones that are near the preselected, but aren't selected
        options = [option for option in self.option_dict.keys() if option not in preselected_bones_indexes]
        # else:
        #     options = list(np.asarray(self.atlas.lines))
        selected_bone = random.choice(options)

        # index = np.where(np.asarray(self.atlas.lines) == selected_bone)[0][0]
        new_bone = o3d.geometry.LineSet()
        new_bone.points = o3d.utility.Vector3dVector(self.atlas.get_line_coordinate(selected_bone))
        new_bone.lines = o3d.utility.Vector2iVector([[0, 1]])
        return new_bone, selected_bone#index

    def load_image(self, path):
        pcd = o3d.io.read_point_cloud(path)

        numpy_source = np.asarray(pcd.points)
        # numpy_source = numpy_source[numpy_source[:, 2] > 400]
        # numpy_source = numpy_source[numpy_source[:, 2] < 601]

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
        bone_points = np.array( [[336, 257, 453],
                                 [275, 173, 466],
                                 [ 64, 271, 473],
                                 [ 27, 232, 471],
                                 [163, 298, 553],
                                 [ 95, 228, 556],
                                 [356, 309, 521],
                                 [375, 214, 509],
                                 [450, 228, 463],
                                 [415, 176, 447],
                                 [148, 267, 486],
                                 [113, 208, 505],
                                 [460, 173, 538],
                                 [427, 106, 522],
                                 [215, 214, 462],
                                 [245, 160, 466]])
        connection = np.arange(bone_points.shape[0]).reshape(-1,2)
        self.rbfs = [
            {'mean': np.array([298.29530201, 204.56375839, 460.02684564]),
              'std': np.array([19.58272878, 25.13168597, 4.16270806])},
            {'mean': np.array([41.68888889, 250.75555556, 469.26666667]),
              'std': np.array([8.26726402, 9.3622304, 2.39814743])},
            {'mean': np.array([121.67379679, 254.06417112, 551.24064171]),
              'std': np.array([23.10278146, 30.44528975, 6.13405115])},
            {'mean': np.array([363.70642202, 255.9266055, 519.5412844]),
              'std': np.array([3.14862065, 15.39492733, 2.0788675])},
            {'mean': np.array([430.07222222, 194.52777778, 451.53333333]),
              'std': np.array([9.01667009, 12.33127485, 8.80933848])},
            {'mean': np.array([125.03875969, 241.95348837, 498.12403101]),
              'std': np.array([8.03037171, 14.10473584, 4.66391018])},
            {'mean': np.array([457.92105263, 145.43421053, 531.81578947]),
              'std': np.array([5.03086044, 10.71316888, 2.78487705])},
            {'mean': np.array([239.68852459, 171.98360656, 466.49180328]),
              'std': np.array([6.88200548, 13.55437356, 1.62584607])}
        ]
        self.atlas.points = o3d.utility.Vector3dVector(bone_points)
        self.atlas.lines = o3d.utility.Vector2iVector(connection)
        self.atlas.paint_uniform_color([0, 1, 1])

    def _get_bone_clusters(self, pcd, bone_line, cluster_dist=20):

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
                continue
        #
        #     b = lineseg_dists(np.asarray(pcd.points)[cluster],
        #                       np.mean(np.asarray(pcd.points)[cluster], axis=0) + bone_line / 2,
        #                       np.mean(np.asarray(pcd.points)[cluster], axis=0) - bone_line / 2, True)
        #     a = lineseg_dists(np.asarray(pcd.points)[cluster],
        #                       np.mean(np.asarray(pcd.points)[cluster], axis=0) + bone_line / 2,
        #                       np.mean(np.asarray(pcd.points)[cluster], axis=0) - bone_line / 2, False)
        #     if np.mean(b-a,axis=0)>1:
        #         continue

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

if __name__ == "__main__":
    hs1 = HyperSkeleton()
    hs1.scan(r"D:\visceral\full_skeletons\102865_CT_Wb.ply")
    hs2 = HyperSkeleton()
    hs2.scan(r"D:\visceral\full_skeletons\102945_CT_Wb.ply")
    transform = calculate_deformation(hs1,hs2)
    hs1.pcd.translate(transform)
    hs1.pcd.paint_uniform_color([1,0,0])
    hs2.pcd.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([hs1.pcd,hs2.pcd])
    # hs.vis_all_results()
