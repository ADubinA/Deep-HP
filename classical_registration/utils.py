import numpy as np
import json

import scipy
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from tqdm import tqdm
import random
from scipy import spatial

# """
# https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/#disqus_thread
# """
# def generate_circle_by_vectors(t, C, r, n, u):
#     n = n / np.linalg.norm(n)
#     u = u / np.linalg.norm(u)
#     P_circle = r * np.cos(t)[:, np.newaxis] * u + r * np.sin(t)[:, np.newaxis] * np.cross(n, u) + C
#     return P_circle
#
#
# def generate_circle_by_angles(t, C, r, theta, phi):
#     # Orthonormal vectors n, u, <n,u>=0
#     n = np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])
#     u = np.array([-np.sin(phi), np.cos(phi), 0])
#
#     # P(t) = r*cos(t)*u + r*sin(t)*(n x u) + C
#     P_circle = r * np.cos(t)[:, np.newaxis] * u + r * np.sin(t)[:, np.newaxis] * np.cross(n, u) + C
#     return P_circle
#
# def fit_circle_2d(x, y, w=[]):
#     A = np.array([x, y, np.ones(len(x))]).T
#     b = x ** 2 + y ** 2
#
#     # Modify A,b for weighted least squares
#     if len(w) == len(x):
#         W = np.diag(w)
#         A = np.dot(W, A)
#         b = np.dot(W, b)
#
#     # Solve by method of least squares
#     c = np.linalg.lstsq(A, b, rcond=None)[0]
#
#     # Get circle parameters from solution c
#     xc = c[0] / 2
#     yc = c[1] / 2
#     r = np.sqrt(c[2] + xc ** 2 + yc ** 2)
#     return xc, yc, r
#
#
# def rodrigues_rot(P, n0, n1):
#     # If P is only 1d array (coords of single point), fix it to be matrix
#     if P.ndim == 1:
#         P = P[np.newaxis, :]
#
#     # Get vector of rotation k and angle theta
#     n0 = n0 / np.linalg.norm(n0)
#     n1 = n1 / np.linalg.norm(n1)
#     k = np.cross(n0, n1)
#     k = k / np.linalg.norm(k)
#     theta = np.arccos(np.dot(n0, n1))
#
#     # Compute rotated points
#     P_rot = np.zeros((len(P), 3))
#     for i in range(len(P)):
#         P_rot[i] = P[i] * np.cos(theta) + np.cross(k, P[i]) * np.sin(theta) + k * np.dot(k, P[i]) * (1 - np.cos(theta))
#
#     return P_rot
#
# def angle_between(u, v, n=None):
#     if n is None:
#         return np.arctan2(np.linalg.norm(np.cross(u,v)), np.dot(u,v))
#     else:
#         return np.arctan2(np.dot(n,np.cross(u,v)), np.dot(u,v))
#
# def calculate_curvature(points):
#
#     points_mean = points.mean(axis=0)
#     points_centered = points - points_mean
#     U,s,V = np.linalg.svd(points_centered)
#
#     # Normal vector of fitting plane is given by 3rd column in V
#     # Note linalg.svd returns V^T, so we need to select 3rd row from V^T
#     normal = V[2,:]
#
#     # Project points to coords X-Y in 2D plane
#     points_xy = rodrigues_rot(points_centered, normal, [0,0,1])
#
#     # Fit circle in new 2D coords
#     try:
#         _, _, r = fit_circle_2d(points_xy[:,0], points_xy[:,1])
#     except np.linalg.LinAlgError:
#         return 0
#     return r

def calculate_curvature(points):
    grad1 = np.gradient(points, axis=0)
    grad2 = np.gradient(grad1, axis=0)
    grad1_norm = np.linalg.norm(grad1, axis=1)
    grad2_norm = np.linalg.norm(grad2, axis=1)

    curvature = np.sqrt((grad1_norm ** 2) * (grad2_norm ** 2) - (grad1 * grad2).sum(axis=1) ** 2) / (grad1_norm ** 3)
    curvature = curvature[2:-2]
    return curvature.mean()

def gaussian_wasserstein_dist(mean1,mean2, cov1,cov2):
    mean_loss = np.linalg.norm(mean1-mean2)
    cov2_sqrt = scipy.linalg.sqrtm(cov2)
    trace_loss = np.trace(cov1 + cov2 - 2 * scipy.linalg.sqrtm(np.matmul(np.matmul(cov2_sqrt,cov1),cov2_sqrt)))
    return np.sqrt(mean_loss + np.real(trace_loss))


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


def graph_contraction(source_graph, contraction_percent):
    g = source_graph.copy()
    max_nodes = int(len(source_graph.nodes) * (1 - contraction_percent))
    pbar = tqdm(total=100)
    while g.number_of_nodes() > max_nodes:
        random_node = random.sample(g.nodes,1)[0]
        nei = list(g.neighbors(random_node))
        if not nei:
            continue
        nx.contracted_nodes(g,random_node,nei[0],self_loops=False,copy=False)
        pbar.set_description(f"num of nodes {g.number_of_nodes()}, trying to reduce to {max_nodes}")
    pbar.close()

    return g

def better_graph_contraction(source_graph, sample_num, max_distance):
    new_nodes = random.sample(source_graph.nodes, sample_num)
    new_graph = nx.Graph()
    new_graph.add_nodes_from([(node,source_graph.nodes[node]) for node in new_nodes])
    kdtree = spatial.KDTree(np.array([source_graph.nodes[new_node]["point"] for new_node in new_nodes]))
    pairs = list(kdtree.query_pairs(max_distance))

    pairs = [(new_nodes[pair[0]],new_nodes[pair[1]]) for pair in pairs]

    pair_dict = {}
    for pair in pairs:
        if pair[0] not in pair_dict:
            pair_dict[pair[0]] = [pair[1]]
        else:
            pair_dict[pair[0]].append(pair[1])

    for pair, option_list in tqdm(pair_dict.items()):
        try:
            length = nx.single_source_shortest_path_length(source_graph,pair, cutoff=max_distance) #nx.single_source_dijkstra_path_length(source_graph,pair, cutoff=max_distance)
            for option in option_list:
                if length.get(option, 2*max_distance) < max_distance:
                    new_graph.add_edge(pair,option, weight=length[option])

                    if not new_graph.nodes[pair].get("contraction", False):
                        new_graph.nodes[pair]["contraction"] = [source_graph.nodes[option]]
                    else:
                        new_graph.nodes[pair]["contraction"].append(source_graph.nodes[option])
                    new_graph.nodes[pair]["contraction"].extend(source_graph.nodes[option].get("contraction",[]))

        except nx.NetworkXNoPath:
            continue

    return new_graph


if __name__ == "__main__":
    i = np.arange(50) * np.pi / 50 - np.pi / 2
    points = np.transpose(np.array([np.cos(0) * np.sin(i), np.sin(0) * np.sin(i), np.cos(i)]))
    print(calculate_curvature(points))
    print(calculate_curvature_mean(points))