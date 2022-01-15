import networkx
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from mayavi import mlab
from create_atlas import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt




def network_plot_3D(G, angle):

    # Get node positions
    pos = networkx.get_node_attributes(G, 'point')

    # Get number of nodes
    n = G.number_of_nodes()

    # 3D network plot
    with plt.style.context(('ggplot')):

        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(xi, yi, zi, s=20 + 20 * G.degree(key), edgecolors='k', alpha=0.7)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in enumerate(G.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)

    # Set the initial view
    ax.view_init(30, angle)

    # Hide the axes
    ax.set_axis_off()
    plt.show()


def create_graph(pcd, graph_point_distance):
    g = networkx.Graph()
    for point_index in tqdm.tqdm(range(np.asarray(pcd.points).shape[0])):
        g.add_node(point_index, point=np.asarray(pcd.points)[point_index]
                              , normal=np.asarray(pcd.normals)[point_index])

    bone_tree = o3d.geometry.KDTreeFlann(pcd)
    for point_index in tqdm.tqdm(range(np.asarray(pcd.points).shape[0])):
        [k, nie, _] = bone_tree.search_radius_vector_3d(pcd.points[point_index], graph_point_distance)

        for element in nie:
            if element == point_index:
                continue
            g.add_edge(point_index, element, weight=np.linalg.norm(np.asarray(pcd.points)[point_index]-np.asarray(pcd.points)[element]))

    return g

def test8_non_atlas_graph():
    normal_rad = 10
    graph_point_distance = 4
    min_path_lengths = 5
    max_path_lengths = 10
    num_bins = 8
    n_clusters = 20
    min_z,max_z = 50,100
    source = o3d.io.read_point_cloud(r"D:\datasets\nmdid\clean-body-pcd\case-100114_BONE_TORSO_3_X_3.ply")
    source,_,_ = source.voxel_down_sample_and_trace(graph_point_distance,min_bound=np.array([0,0,min_z]),max_bound=np.array([1000,1000,max_z]))
    source.paint_uniform_color([0.1, 0.1, 0.1])
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_rad, max_nn=30))

    g = create_graph(source,graph_point_distance)
    # g = graph_contraction(g,0.5)
    dix = networkx.all_pairs_shortest_path(g,max_path_lengths)
    # dix = networkx.all_pairs_dijkstra(g, max_path_lengths)
    bins_list = []
    for point_index,paths in tqdm.tqdm(dix, total=g.number_of_nodes()):
        # paths = paths[1]  # get only the paths, not the lens
        paths = [path for _, path in paths.items() if len(path) > min_path_lengths]
        if len(paths) < 5:
            bins = np.zeros((num_bins,num_bins))
        else:
            path_edges = [g.nodes[path[-1]]["point"] for path in paths]
            dist = np.asarray(path_edges) - g.nodes[point_index]["point"]
            dist = to_spherical(dist)
            bins,_,_ = np.histogram2d(dist[:,1],dist[:,2],np.arange(num_bins+1)*1/num_bins, density=True, range=(0,1))

            bins = bins/bins.sum()
        g.nodes[point_index]["histogram"] = bins
        bins_list.append(bins)

    bins_list = np.stack(bins_list)
    flat_bins = bins_list.reshape(bins_list.shape[0], -1)
    k_means = KMeans(init="k-means++", n_clusters=n_clusters, n_init=5)
    results = k_means.fit(flat_bins)

    for node in g.nodes:
        g.nodes[node]['label'] = results.labels_[node]
    different_cluster_edges = [edge for edge in g.edges if g.nodes[edge[0]]["label"] != g.nodes[edge[1]]["label"]]
    g.remove_edges_from(different_cluster_edges)

    # histogram mapping colors
    # colors = np.random.random((results.labels_.shape[0],3))
    # np.asarray(source.colors)[np.array(list(g.nodes))] = colors[results.labels_]
    # o3d.visualization.draw([source])

    # gaussian points visualization
    for connected in networkx.connected_components(g):
        if len({g.nodes[node]["label"] for node in connected})!=1:
            raise ValueError("bad.")
        color = np.random.random([3])
        np.asarray(source.colors)[list(connected)] = color
        # points = networkx.get_node_attributes(g,"point")
        # points = points[connected]
    o3d.visualization.draw([source])




if __name__ == "__main__":
    test8_non_atlas_graph()