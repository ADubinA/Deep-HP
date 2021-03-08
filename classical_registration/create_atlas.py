# examples/Python/Advanced/interactive_visualization.py

import os, glob
import numpy as np
import copy
import open3d as o3d
import tqdm
import random
from sklearn.cluster import AgglomerativeClustering

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
    h = np.expand_dims(h,axis = -1)
    # perpendicular distance component, as before
    # note that for the 3D case these will be vectors
    c = np.cross(p - a, d)

    # use hypot for Pythagoras to improve accuracy
    return np.sqrt(np.sum(np.power(np.hypot(h, c),2), axis=-1))

def to_spherical(points, normalize_angles=True):
    points_spherical = np.zeros_like(points)
    xy_norm = points[:,0]**2 + points[:,1]**2

    points_spherical[:,0] = np.sqrt(xy_norm + points[:,2]**2) # radius
    points_spherical[:,1] = np.arctan2(np.sqrt(xy_norm), points[:,2]) # for elevation angle defined from Z-axis down
    points_spherical[:,2] = np.arctan2(points[:,1], points[:,0])

    if normalize_angles:
        points_spherical[:, 0] = points_spherical[:,0]/points_spherical[:,0].max()
        points_spherical[:, 1] = (points_spherical[:,1])/(np.pi)
        points_spherical[:, 2] = (points_spherical[:, 2] + np.pi) / (2 * np.pi)
    return points_spherical

def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print(vis.get_picked_points())
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
    o3d.visualization.draw_geometries([pcd])

def get_bone_groups(pcd, bone_line, cluster_dist = 20):
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
        if (dists < 1).sum() > np.linalg.norm(bone_line) / 10:
            np.asarray(pcd.colors)[point_index] = [1, 0, 0]
            choosen_points.append(np.asarray(pcd.points)[point_index])
            choosen_indexes.append(point_index)

    choosen_points = np.array(choosen_points)
    choosen_indexes = np.array(choosen_indexes)
    clustering = AgglomerativeClustering(n_clusters=None,distance_threshold=cluster_dist, linkage="single").fit(choosen_points)
    colors = np.random.random((clustering.labels_.shape[0],3))
    np.asarray(pcd.colors)[choosen_indexes] = colors[clustering.labels_]
    o3d.visualization.draw_geometries([pcd])
def test1_get_some_bones():
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
    points = pick_points(source)
    bone_line = np.asarray(source.points)[points[0]] - np.asarray(source.points)[points[1]]
    print(bone_line)
    pcd = remove_non_normals(source,bone_line)
    check_full_bone(pcd,bone_line)
    # pcd = remove_interior(source)
    # remove_non_normals(pcd, bone_line)
def test2_legs():
    skeleton_folder = r"D:\visceral\full_skeletons"
    # bone_line = np.array([84,170,450]) - np.array([84,82,400]) # Legs
    # bone_line = np.array([390,320,540]) - np.array([420,260,560]) # left pelvis bone
    bone_line = np.array([-70. , 46. , 26.]) # right front pelvis bone
    skeleton_paths = random.sample(glob.glob(os.path.join(skeleton_folder,"*.ply")), 10)

    for file_name in skeleton_paths:
        pcd = o3d.io.read_point_cloud(file_name)
        o3d.visualization.draw_geometries([pcd])
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
    get_bone_groups(source, bone_line)
if __name__ == "__main__":
    # test1_get_some_bones()
    # test2_legs()
    test3_clustering()