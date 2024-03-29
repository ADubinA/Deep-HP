# examples/Python/Advanced/interactive_visualization.py
import time

import networkx
import numpy as np
import copy
import open3d as o3d
import tqdm
import random
from sklearn.cluster import AgglomerativeClustering
import glob, os
from scipy.spatial.transform import Rotation as R
import json

def histogram_features(pcd,rad = 10,min_var =0.085):
    pcd.paint_uniform_color([0.0, 0.5, 0.5])
    num_bins = 8
    min_var = 0.08
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    features = np.array([[]])
    good_indexes = []

    for point_index in tqdm.tqdm(range(np.asarray(pcd.points).shape[0])):

        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[point_index], rad)
        if np.asarray(idx).shape[0]<10:
            np.asarray(pcd.colors)[point_index] = np.array([0,0,0])
            continue

        dist = np.asarray(pcd.points)[idx[1:],:] - np.asarray(pcd.points)[idx[0]]
        dist = to_spherical(dist)
        dist[:,0] = 0
        if dist.var()>min_var:
            np.asarray(pcd.colors)[point_index] = [0, 0, 1]
            continue

        good_indexes.append(point_index)

        bins = np.histogram2d(dist[:,1],dist[:,2],np.arange(num_bins)*1/num_bins, density=True, range=(0,1))

        # try:
        #     features = np.concatenate((features, np.expand_dims(bins[0], axis=0)))
        # except ValueError:
        #     features = np.expand_dims(bins[0], axis=0)

        max_bin = np.unravel_index(np.argmax(bins[0], axis=None), bins[0].shape)
        np.asarray(pcd.colors)[point_index] = [bins[1][max_bin[0]],bins[2][max_bin[1]],0]

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[good_indexes])
    new_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[good_indexes])
    new_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[good_indexes])
    # num_cluster=8*8
    # prediction = KMeans(n_clusters=num_cluster, random_state=0).fit_predict(np.asarray(pcd.colors))
    # colors_idx = np.random.random((num_cluster,3))
    # pcd.colors = o3d.utility.Vector3dVector(colors_idx[prediction])
    return features, new_pcd
def lineseg_dists(p, a, b, clamp=True):

    # TODO for you: consider implementing @Eskapp's suggestions
    if clamp:
        if np.all(a == b):
            return np.linalg.norm(p - a, axis=1)

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
        return np.linalg.norm(np.cross(b-a, a-p))/np.linalg.norm(b-a)

def to_spherical(points, normalize_angles=True):
    points_spherical = np.zeros_like(points)
    xy_norm = points[:,0]**2 + points[:,1]**2

    points_spherical[:,0] = np.sqrt(xy_norm + points[:,2]**2) # radius
    points_spherical[:,1] = np.arctan2(np.sqrt(xy_norm), points[:,2]) # for elevation angle defined from Z-axis down
    points_spherical[:,2] = np.arctan2(points[:,1], points[:,0])

    if normalize_angles:
        points_spherical[:, 0] = points_spherical[:,0]
        points_spherical[:, 1] = (points_spherical[:,1])/(np.pi)
        points_spherical[:, 2] = (points_spherical[:, 2] + np.pi) / (2 * np.pi)
    return points_spherical

def pick_points(pcd):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()

    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print(np.asarray(pcd.points)[vis.get_picked_points()])
    return vis.get_picked_points()

def remove_interior(pcd,rad = 15):
    pcd.paint_uniform_color([0.0, 0.5, 0.5])
    num_bins = 8
    min_var = 0.1
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    indexes = []
    for point_index in tqdm.tqdm(range(np.asarray(pcd.points).shape[0])):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[point_index], rad)
        if np.asarray(idx).shape[0]<10:
            np.asarray(pcd.colors)[point_index] = np.array([0,0,0])
            continue
        dist = np.asarray(pcd.points)[idx[3:],:] - np.asarray(pcd.points)[point_index]
        dist = to_spherical(dist)
        dist[:,0] = 0
        if dist.var()>min_var:
            indexes.append(point_index)
        # bins = np.histogram2d(dist[:,1],dist[:,2],np.arange(num_bins)*1/num_bins, density=True)
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[indexes])
    new_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[indexes])
    new_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[indexes])
    o3d.visualization.draw_geometries([new_pcd])
    return new_pcd

def remove_non_normals(pcd,bone_line, rad=20):
    bone_line =bone_line/np.linalg.norm(bone_line,axis=-1)
    products = np.asarray(pcd.normals).dot(bone_line)
    bone_pcd = o3d.geometry.PointCloud()
    idx = np.abs(products)<0.01
    bone_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[idx])
    bone_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[idx])
    bone_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[idx])


    # bone_tree = o3d.geometry.KDTreeFlann(bone_pcd)
    # for point_index in range(np.asarray(bone_pcd.points).shape[0]):
    #     [k, nie, _] = bone_tree.search_radius_vector_3d(bone_pcd.points[point_index], rad)
    #     if len(nie)<5:
    #         continue
    #     dist = np.asarray(bone_pcd.points)[nie[1:], :] - np.asarray(bone_pcd.points)[point_index]
    #     dist = dist/np.expand_dims(np.linalg.norm(dist,axis=-1),axis=-1)
    #     point_fitness = np.abs((np.abs(dist) - np.abs(bone_line))).sum(axis=-1).mean()
    #
    #     if point_fitness<0.5:
    #         np.asarray(bone_pcd.colors)[point_index] = [1,0,0]
    # o3d.visualization.draw_geometries([bone_pcd])
    return bone_pcd

def remove_non_on_tangent(pcd,bone_line, rad=10):
    bone_tree = o3d.geometry.KDTreeFlann(pcd)
    for point_index in tqdm.tqdm(range(np.asarray(pcd.points).shape[0])):
        [k, nie, _] = bone_tree.search_radius_vector_3d(pcd.points[point_index], rad)
        if len(nie)<5:
            continue
        dist = np.asarray(pcd.points)[nie[1:], :] - np.asarray(pcd.points)[point_index]
        dist = dist/np.expand_dims(np.linalg.norm(dist,axis=-1),axis=-1)
        point_fitness = np.abs((np.abs(dist) - np.abs(bone_line))).sum(axis=-1).min()
        if point_fitness is np.nan:
            print("nan")
            continue
        if point_fitness<0.1:
            np.asarray(pcd.colors)[point_index] = [1,0,0]

    o3d.visualization.draw_geometries([pcd])
    return pcd
def check_full_bone(pcd,bone_line):

    bone_tree = o3d.geometry.KDTreeFlann(pcd)
    for point_index in tqdm.tqdm(range(np.asarray(pcd.points).shape[0])):
        [k, nie, _] = bone_tree.search_radius_vector_3d(pcd.points[point_index], np.linalg.norm(bone_line)/2)
        if len(nie)<5:
            continue

        dists = lineseg_dists(np.asarray(pcd.points)[nie[1:],:],
                      np.asarray(pcd.points)[point_index] + bone_line/2,
                      np.asarray(pcd.points)[point_index] - bone_line/2)
        if (dists<1).sum()>5:
            np.asarray(pcd.colors)[point_index] = [1,0,0]
    return pcd
    # o3d.visualization.draw_geometries([pcd])

def get_bone_clusters(pcd, bone_line, cluster_dist = 5):
    bone_tree = o3d.geometry.KDTreeFlann(pcd)
    # pcd.paint_uniform_color((0,0,0))
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
            np.asarray(pcd.colors)[point_index] = np.array([1, 0, 0])
            choosen_points.append(np.asarray(pcd.points)[point_index])
            choosen_indexes.append(point_index)

    if len(choosen_indexes) <2:
        return pcd, []
    choosen_points = np.array(choosen_points)
    choosen_indexes = np.array(choosen_indexes)
    clustering = AgglomerativeClustering(n_clusters=None,distance_threshold=cluster_dist, linkage="single").fit(choosen_points)

    colors = np.random.random((clustering.labels_.shape[0],3))
    np.asarray(pcd.colors)[choosen_indexes] = colors[clustering.labels_]

    clusters =[]
    for cluster_index in range(clustering.n_clusters_):
        cluster = choosen_indexes[clustering.labels_== cluster_index]
        if cluster.shape[0]<5:
            np.asarray(pcd.colors)[cluster] = np.zeros(3)
            continue
        if np.asarray(pcd.points)[cluster].std(axis=0).max()<10:
            np.asarray(pcd.colors)[cluster] = np.zeros(3)
            continue
        clusters.append(cluster)


    # print(len(choosen_points))
    return pcd, clusters

def print_possible_bone(pcd, clusters_indexes, bone_line, previous_lines):
    true_bone_option = []
    for cluster in clusters_indexes:
        points = np.asarray(pcd.points)[cluster]
        mean_point = points.mean(axis=0)
        new_bone_line = bone_line - (bone_line[1]-bone_line[0])/2 - bone_line[0]  + mean_point # center point
        true_bone_option.append(new_bone_line)

    if previous_lines == []:
        return true_bone_option
    u = true_bone_option[:, None] - previous_lines
    best_option = np.unravel_index(np.linalg.norm(u, axis=2).argmin(), np.linalg.norm(u, axis=2).shape)
    return previous_lines[best_option[1]], true_bone_option[best_option[0],:]
def test1_get_some_bones():
    normal_rad = 10
    source = o3d.io.read_point_cloud(r"D:\visceral\full_skeletons\102946_CT_Wb.ply")
    numpy_source = np.asarray(source.points)
    numpy_source = numpy_source[numpy_source[:,2]>300]
    numpy_source = numpy_source[numpy_source[:, 2]<501]
    resample = 0.75
    source_index = np.random.randint(0,numpy_source.shape[0], int(numpy_source.shape[0]*resample))
    source.points = o3d.utility.Vector3dVector(numpy_source[source_index])
    source.paint_uniform_color([0.1, 0.1, 0.1])
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_rad, max_nn=30))

    bone_lineset = o3d.geometry.LineSet()
    bone_lineset.points = source.points
    bone_lineset.lines = o3d.utility.Vector2iVector(np.array([[245884, 68646]]))
    bone_lineset.paint_uniform_color([1,0,0])

    points = pick_points(source)
    bone_line = np.asarray(source.points)[points[0]] - np.asarray(source.points)[points[1]]
    print(bone_line)
    pcd = remove_non_normals(source,bone_line)
    pcd = check_full_bone(pcd,bone_line)

    o3d.visualization.draw_geometries(([pcd,bone_lineset]))
    # pcd = remove_interior(source)
    # remove_non_normals(pcd, bone_line)
def test2_legs():
    skeleton_folder = r"D:\visceral\full_skeletons"
    # bone_line = np.array([84,170,450]) - np.array([84,82,400]) # Legs
    bone_line = np.array([200,340,540]) - np.array([90,220,550]) # left pelvis bone
    # bone_line = np.array([-70. , 46. , 26.]) # right front pelvis bone
    bone_lineset = o3d.geometry.LineSet()
    bone_lineset.points = o3d.utility.Vector3dVector(np.array([[390,320,540],np.array([420,260,560])]))
    bone_lineset.lines = o3d.utility.Vector2iVector(np.array([[0,1]]))
    skeleton_paths = random.sample(glob.glob(os.path.join(skeleton_folder,"*.ply")), 10)

    for file_name in skeleton_paths:
        pcd = o3d.io.read_point_cloud(file_name)
        o3d.visualization.draw_geometries([pcd,bone_lineset])
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
        pcd = remove_non_normals(pcd, bone_line)
        check_full_bone(pcd, bone_line)

def test3_clustering():
    normal_rad = 10
    source = o3d.io.read_point_cloud(r"D:\visceral\full_skeletons\102946_CT_Wb.ply")
    numpy_source = np.asarray(source.points)
    numpy_source = numpy_source[numpy_source[:,2]>300]
    numpy_source = numpy_source[numpy_source[:, 2]<801]
    resample = 0.75
    source_index = np.random.randint(0,numpy_source.shape[0], int(numpy_source.shape[0]*resample))
    source.points = o3d.utility.Vector3dVector(numpy_source[source_index])
    source.paint_uniform_color([0.1, 0.1, 0.1])
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_rad, max_nn=30))

    bone_line = np.array([-70., 46., 26.])  # right front pelvis bone
    source = remove_non_normals(source, bone_line)
    pcd = get_bone_clusters(source, bone_line)


def test4_test_random_dir():
    normal_rad = 10
    sample_size = 20
    source = o3d.io.read_point_cloud(r"D:\visceral\full_skeletons\102946_CT_Wb.ply")
    numpy_source = np.asarray(source.points)
    numpy_source = numpy_source[numpy_source[:,2]>300]
    numpy_source = numpy_source[numpy_source[:, 2]<501]
    resample = 0.75
    source_index = np.random.randint(0,numpy_source.shape[0], int(numpy_source.shape[0]*resample))
    source.points = o3d.utility.Vector3dVector(numpy_source[source_index])
    source.paint_uniform_color([0.1, 0.1, 0.1])
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_rad, max_nn=30))

    r = R.random(sample_size)
    vector_length = np.random.randint(20,50, (sample_size,3))
    vector_length[:,:2] = 0
    bones = r.apply(vector_length)
    for bone in bones:
        bone_line =  o3d.geometry.LineSet()
        bone_line.points = o3d.utility.Vector3dVector([[0,0,500],bone + np.array([0,0,500])])
        bone_line.lines = o3d.utility.Vector2iVector([[0,1]])
        bone_line.paint_uniform_color([0,1,0])
        print(bone)
        pcd = o3d.geometry.PointCloud(source)
        pcd = remove_non_normals(pcd, bone)
        pcd = get_bone_clusters(pcd, bone)
        np.asarray(pcd.colors)
        o3d.visualization.draw_geometries([pcd,bone_line])

def test5_atlas():
    bone_points = np.array([[181.76923077, 313.03846154,550.23076923],
                            [ 79.4875,     184.775,      542.45625   ],
                            [126.25   ,    163.5   ,     515.75      ],
                            [179.00625 ,   185.44791667, 497.09345238],
                            [239.66666667, 131.33333333 ,476.66666667],
                            [237.24444444,193.60520833, 465.11875   ],
                            [183.13333333, 275.72982456, 472.12982456]])
    connection = np.array([[0,1],[1,4],[4,5],[5,6]])
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(bone_points)
    lineset.lines = o3d.utility.Vector2iVector(connection)
    lineset.paint_uniform_color([0,0,0])
    normal_rad = 10
    sample_size = 20
    source = o3d.io.read_point_cloud(r"D:\visceral\full_skeletons\102946_CT_Wb.ply")
    source.paint_uniform_color([0.1, 0.1, 0.1])
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_rad, max_nn=30))
    previous_options = []
    for bone_index in range(6):
        np.asarray(lineset.colors)[bone_index] = np.array([1,0,0])
        bone_line = np.array(bone_points[bone_index] - bone_points[bone_index+1])
        pcd = o3d.geometry.PointCloud(source)
        pcd = remove_non_normals(pcd, bone_line)
        pcd, clusters_index = get_bone_clusters(pcd, bone_line)
        # o3d.visualization.draw_geometries([lineset,pcd])
        # previous_options= [print_possible_bone(pcd,clusters_index,bone_points[bone_index:bone_index+2,:],previous_options)[1]]
        #
        # lineset = o3d.geometry.LineSet()
        # lineset.points = o3d.utility.Vector3dVector(previous_options[-1])
        # lineset.lines = o3d.utility.Vector2iVector(np.array([[0,1]]))
        # selected.append(lineset)

        o3d.visualization.draw_geometries([pcd, lineset])

def test6_histograms_atlas():
    pcd = o3d.io.read_point_cloud(r"D:\visceral\full_skeletons\102946_CT_Wb.ply")
    numpy_source = np.asarray(pcd.points)
    numpy_source = numpy_source[numpy_source[:, 2] > 400]
    numpy_source = numpy_source[numpy_source[:, 2] < 601]
    resample = 0.1
    normal_rad = 20
    pcd_index = np.random.randint(0, numpy_source.shape[0], int(numpy_source.shape[0] * resample))
    pcd.points = o3d.utility.Vector3dVector(numpy_source[pcd_index])
    pcd.paint_uniform_color([0.1, 0.1, 0.1])
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_rad, max_nn=30))

    _, new_pcd = histogram_features(pcd)
    # new_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_rad, max_nn=30))
    good_lines = []
    gausses = []
    while True:
        points = pick_points(new_pcd)

        if len(points)<2:
            good_lines.pop(-1)
            gausses.pop(-1)
            continue
        if len(points)>3:
            print("resulting lines")
            for good_line in good_lines:
                print(np.asarray(good_line.points))
            print("resulting gausses")
            for gauss in gausses:
                print(gauss)
        bone_line = np.asarray(new_pcd.points)[points[0]] - np.asarray(new_pcd.points)[points[1]]
        print(bone_line)

        # pcd = remove_non_normals(new_pcd, bone_line)
        pcd, clusters_index = get_bone_clusters(new_pcd.__copy__(), bone_line)

        # find the best cluster for the line (spataily)
        best_dist = float("inf")
        best_cluster = None
        for cluster in clusters_index:
            dist = lineseg_dists(np.asarray(pcd.points)[cluster],
                          np.asarray(new_pcd.points)[points[0]],
                          np.asarray(new_pcd.points)[points[1]])
            if np.sum((np.where(dist<0.1)))>20:
                best_cluster = cluster
                break
            if best_dist>dist.mean(axis=0):
                best_dist = dist.mean(axis=0)
                best_cluster = cluster

        gausses.append({"mean":np.asarray(pcd.points)[best_cluster].mean(axis=0),
                   "std":np.asarray(pcd.points)[best_cluster].std(axis=0)})
        pcd.paint_uniform_color((0,0,0))
        np.asarray(pcd.colors)[best_cluster] = np.array([1,0,0])
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector([np.asarray(new_pcd.points)[points[0]],
                                                     np.asarray(new_pcd.points)[points[1]]])
        lineset.lines = o3d.utility.Vector2iVector(np.array([[0,1]]))
        good_lines.append(lineset)
        o3d.visualization.draw_geometries([pcd]+good_lines)
def test7_minimal_direction_graph():
    line_size = 30
    pcd = o3d.io.read_point_cloud(r"D:\datasets\nmdid\pcd\127310_BONE_TORSO_3_X_3.ply")
    save_path = r"D:\datasets\nmdid\experiments\atlases\atlas_1.json"
    summery = "22-nov-first try with nmdid"
    numpy_source = np.asarray(pcd.points)
    # numpy_source = numpy_source[numpy_source[:, 2] > 400]
    # numpy_source = numpy_source[numpy_source[:, 2] < 601]
    resample = 1
    normal_rad = 10
    pcd_index = np.random.randint(0, numpy_source.shape[0], int(numpy_source.shape[0] * resample))
    pcd.points = o3d.utility.Vector3dVector(numpy_source[pcd_index])
    pcd.paint_uniform_color([0.1, 0.1, 0.1])
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_rad, max_nn=30))

    _, new_pcd = histogram_features(pcd)
    new_pcd.paint_uniform_color((0,0,0))
    good_lines = []
    gausses = []
    pcds=[new_pcd.__copy__()]
    while True:
        points = pick_points(pcds[-1])
        if len(points) < 2:
            good_lines.pop(-1)
            gausses.pop(-1)
            pcds.pop(-1)
            continue
        if len(points) > 2:
            print("resulting lines")
            for good_line in good_lines:
                print(np.asarray(good_line.points))
            print("resulting gausses")
            for gauss in gausses:
                print(gauss)

            with open(save_path, 'w') as f:
                num_lines = 0
                for gauss in gausses:
                    num_lines += len(gauss)
                j = {"atlas": {str(np.asarray(good_lines[i].points)[1].tolist()):gausses[i] for i in range(len(good_lines))}}
                j["num_lines"] = num_lines
                j["summery"] = summery
                json.dump(j, f)
            break
        # bone_line = np.random.rand(3)-1/2
        # bone_line = line_size*bone_line/np.linalg.norm(bone_line)
        bone_line = np.asarray(new_pcd.points)[points[0]] - np.asarray(new_pcd.points)[points[1]]
        pcd, clusters = get_bone_clusters(pcds[-1].__copy__(), bone_line)
        pcds.append(pcd)
        gausses.append([])
        cover_num = 0
        for cluster in clusters:
            # calculate the cluster index
            cluster_index = 0
            for gauss in gausses:
                cluster_index+= len(gauss)
            gausses[len(good_lines)].append({"mean": (np.asarray(pcd.points)[cluster].mean(axis=0)).tolist(),
                                             "std": (np.asarray(pcd.points)[cluster].std(axis=0)).tolist(),
                                             "color":np.random.random(3).tolist(),
                                             "axis":bone_line.tolist(),
                                             "index": cluster_index })
            cover_num+=len(cluster)
        if cover_num/len(pcd.points)>0.1:
            print("good coverage")
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector([np.zeros((3)),
                                                     bone_line])
        lineset.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
        good_lines.append(lineset)
        o3d.visualization.draw_geometries([pcds[-1]] + good_lines)

def create_rfb(mean, std, axis,color=np.array([0,0,0])):
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh_sphere.transform(np.array([[std[0],0,0,0],
                                    [0,std[1],0,0],
                                    [0,0,std[2],0],
                                    [0,0,0,1]]))
    unit = np.array([1,0,0])
    a, b = (unit / np.linalg.norm(unit)).reshape(3), (axis / np.linalg.norm(axis)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    mesh_sphere.rotate(rotation_matrix,np.array([0,0,0]))
    mesh_sphere.translate(mean)
    mesh_sphere.paint_uniform_color(color.tolist())

    # line = o3d.geometry.LineSet()
    # line.points = o3d.utility.Vector3dVector([[0,0,0],axis])
    # line.lines = o3d.utility.Vector2iVector([[0,1]])
    # o3d.visualization.draw([line,mesh_sphere])
    return mesh_sphere





if __name__ == "__main__":

    # test1_get_some_bones()
    # test2_legs()
    # test3_clqqqustering()
    # test4_test_random_dir()
    # test5_atlas()
    # test6_histograms_atlas()
    test7_minimal_direction_graph()
    # create_rfb(np.array([1,1,0]),np.array([0.1,0.1,3]),np.array([1,1,0]))