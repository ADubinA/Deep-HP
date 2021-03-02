import open3d as o3d
import numpy as np
import copy
import time
from sklearn.decomposition import PCA
import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def compute_curvature(pcd, radius=0.5):

    points = np.asarray(pcd.points)

    from scipy.spatial import KDTree
    tree = KDTree(points)

    curvature = [ 0 ] * points.shape[0]
    print(f"num of points{points.shape[0]}" )
    for index, point in tqdm.tqdm(enumerate(points)):
        indices = tree.query_ball_point(point, radius)
        if len(indices)<5:
            continue
        # local covariance

        M = np.array([ points[i] for i in indices ]).T
        M = np.cov(M)

        # eigen decomposition
        try:
            V, E = np.linalg.eig(M)
            # h3 < h2 < h1
            h1, h2, h3 = V

            curvature[index] = h3 / (h1 + h2 + h3)
        except:
            pass

    return np.array(curvature)

def print_correspondance(source, target, results):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0.5, 0.50, 0.5])
    target_temp.paint_uniform_color([0.7, 0.7, 0.7])
    source_colors = np.asarray(source_temp.colors)
    target_colors = np.asarray(target_temp.colors)

    source_points = np.asarray(source_temp.points)
    target_points = np.asarray(target_temp.points)
    target_num_points = source_points.shape[0]
    total = np.append(source_points,target_points,axis = 0)
    lines = []
    colors = []
    for corr in np.asarray(results.correspondence_set):
        color = np.random.random((3))
        colors.append(color)
        source_colors[corr[0]] = color
        target_colors[corr[1]] = color
        lines.append([corr[0], corr[1] + target_num_points])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(total),
        lines=o3d.utility.Vector2iVector(lines),
    )

    line_set.colors = o3d.utility.Vector3dVector(colors)
    source_temp.colors = o3d.utility.Vector3dVector(source_colors)
    target_temp.colors = o3d.utility.Vector3dVector(target_colors)
    o3d.visualization.draw_geometries([source_temp, target_temp, line_set])

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

def get_features_vis(pcd, fpfh, dim=None):
    pcd_temp = copy.deepcopy(pcd)
    pcd_temp.paint_uniform_color([1, 0.706, 0])

    # features = np.asarray(fpfh.data)
    features = np.moveaxis(np.asarray(fpfh.data), -1, 0)
    # features = features[:,:3]
    if not dim:
        pca = PCA(n_components=3)
        features = pca.fit_transform(features)
    else:
        pca = dim
        features = pca.transform(features)

    feature_norm = features/np.max(features, axis=0)

    # feature_norm[feature_norm<0.5] = 0
    # colors = np.zeros(np.asarray(pcd.points).shape)
    colors = feature_norm
    pcd_temp.colors = o3d.utility.Vector3dVector(colors)

    # o3d.visualization.draw_geometries([pcd_temp])
    return pcd_temp, pca


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    # source = o3d.io.read_point_cloud(r"..\testing\data_open3d\cloud_bin_0.pcd")
    # target = o3d.io.read_point_cloud(r"..\testing\data_open3d\cloud_bin_1.pcd")
    source = o3d.io.read_point_cloud(r"D:\visceral\full_skeletons\102946_CT_Wb.ply")
    # target = o3d.io.read_point_cloud(r"D:\visceral\full_skeletons\102946_CT_Wb.ply")
    #
    # source = o3d.io.read_point_cloud(r"D:\bun_zipper.ply")
    target = o3d.io.read_point_cloud(r"D:\visceral\full_skeletons\102839_CT_Wb.ply")
    numpy_source = np.asarray(source.points)
    numpy_source = numpy_source[numpy_source[:,2]>400]
    numpy_source = numpy_source[numpy_source[:, 2]<501]

    resample = 0.25
    source_index = np.random.randint(0,numpy_source.shape[0], int(numpy_source.shape[0]*resample))
    target_index = np.random.randint(0,len(target.points),int(len(target.points)*resample) )

    source.points = o3d.utility.Vector3dVector(numpy_source[source_index])
    target.points = o3d.utility.Vector3dVector(np.asarray(target.points)[target_index])

    # trans_init = np.asarray([[0.0, 0.0, 1.0, 500.0],
    #                           [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0],
    #                          [0.0, 0.0, 0.0, 1.0]])
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source,0.001)# voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    source_down,pca = get_features_vis(source_down, source_fpfh)
    target_down,_ = get_features_vis(target_down, target_fpfh, pca)
    # o3d.visualization.draw_geometries([source_down])
    # o3d.visualization.draw_geometries([source_down,target_down])
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000)
    )
    return result

def filter_non_feature_points(pcd, thresh=0.6):

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    normals = np.asarray(pcd.normals)
    ind = np.logical_or(colors[:,0]>thresh, colors[:,1]>thresh, colors[:,2]>thresh)
    points = points[ind]
    colors = colors[ind]
    normals = normals[ind]
    tmp_pcd = o3d.geometry.PointCloud()
    tmp_pcd.points = o3d.utility.Vector3dVector(points)
    tmp_pcd.colors = o3d.utility.Vector3dVector(colors)
    tmp_pcd.normals = o3d.utility.Vector3dVector(normals)
    return tmp_pcd
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
def print_tree(pcd, rad=15,filter_size=0.3):
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    for point_index in range(np.asarray(pcd.points).shape[0]):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[point_index], rad)
        if np.asarray(idx).shape[0]<10:
            np.asarray(pcd.colors)[idx[0]] = np.array([0,0,0])
            continue
        dist = np.asarray(pcd.points)[idx[0]] - np.asarray(pcd.points)[idx[1:],:]
        norm_dist = dist/np.linalg.norm(dist,axis=1).max()
        np.asarray(pcd.colors)[idx[0]] = np.abs(norm_dist.mean(axis=0))
        # vis_dist = (norm_dist + 1)/2
    filter_pcd = o3d.geometry.PointCloud()
    idx = np.logical_or(np.asarray(pcd.colors)[:,0]>filter_size,np.asarray(pcd.colors)[:,1]>filter_size, np.asarray(pcd.colors)[:,2]>filter_size)
    filter_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[idx,:])
    filter_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[idx, :])
    filter_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[idx, :])
    return filter_pcd

def histogram_features(pcd,rad = 15):
    pcd.paint_uniform_color([0.0, 0.5, 0.5])
    num_bins = 8
    min_var = 0.08
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    features = np.array([[]])
    for point_index in range(np.asarray(pcd.points).shape[0]):
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
        bins = np.histogram2d(dist[:,1],dist[:,2],np.arange(num_bins)*1/num_bins, density=True)
        try:
            features = np.concatenate((features, np.expand_dims(bins[0], axis=0)))
        except ValueError:
            features = np.expand_dims(bins[0], axis=0)
        max_bin = np.unravel_index(np.argmax(bins[0], axis=None), bins[0].shape)
        np.asarray(pcd.colors)[point_index] = [bins[1][max_bin[0]],bins[2][max_bin[1]],0]

    # num_cluster=8*8
    # prediction = KMeans(n_clusters=num_cluster, random_state=0).fit_predict(np.asarray(pcd.colors))
    # colors_idx = np.random.random((num_cluster,3))
    # pcd.colors = o3d.utility.Vector3dVector(colors_idx[prediction])
    return features, pcd

def find_matching(source, target, source_features,target_features):
    correspondence = None
    for ind in range(source_features.shape[0]):
        feature = source_features[ind]
        losses = np.power(target_features-feature, 2).sum(axis=-1).sum(axis=-1)
        if losses.min()>30:
            continue
        min_idx = losses.argmin()
        try:
            correspondence = np.concatenate((correspondence, np.array([[ind,min_idx]])))
        except ValueError:
            correspondence = np.array([[ind,min_idx]])


    print_correspondence_numpy(source, target, correspondence)

def print_correspondence_numpy(source, target, correspondence):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0.5, 0.50, 0.5])
    target_temp.paint_uniform_color([0.7, 0.7, 0.7])
    source_colors = np.asarray(source_temp.colors)
    target_colors = np.asarray(target_temp.colors)

    source_points = np.asarray(source_temp.points)
    target_points = np.asarray(target_temp.points)
    target_num_points = source_points.shape[0]
    total = np.append(source_points,target_points,axis = 0)
    lines = []
    colors = []
    for corr in correspondence:
        color = np.random.random((3))
        colors.append(color)
        source_colors[corr[0]] = color
        target_colors[corr[1]] = color
        lines.append([corr[0], corr[1] + target_num_points])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(total),
        lines=o3d.utility.Vector2iVector(lines),
    )

    line_set.colors = o3d.utility.Vector3dVector(colors)
    source_temp.colors = o3d.utility.Vector3dVector(source_colors)
    target_temp.colors = o3d.utility.Vector3dVector(target_colors)
    o3d.visualization.draw_geometries([source_temp, target_temp, line_set])
if __name__ == "__main__":
    voxel_size = 5  # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        voxel_size)
    source_his, source_down = histogram_features(source_down)
    target_his, target_down = histogram_features(target_down)
    find_matching(source_down,target_down,source_his,target_his)






    source_down = print_tree(source_down,rad=5)
    target_down = print_tree(target_down)
    o3d.visualization.draw_geometries([source_down,target_down])

    source_down.paint_uniform_color([0,0,0])
    target_down.paint_uniform_color([0,0,0])

    np.asarray(source_down.colors)[:,0] = compute_curvature(source_down,radius=10)
    np.asarray(target_down.colors)[:,0] = compute_curvature(target_down,radius=10)
    o3d.visualization.draw_geometries([source_down,target_down])
    pass
    # result_icp = refine_registration(source_down,target_down,voxel_size)
    # print(result_icp)
    # draw_registration_result(source, target, result_icp.transformation)
    # result_ransac = execute_global_registration(source_down, target_down,
    #                                             source_fpfh, target_fpfh,
    #                                             voxel_size)
    #
    # print(result_ransac)
    # radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    # source.estimate_normals(
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    # radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    # target.estimate_normals(
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    # print_correspondance(source, target, result_ransac)
    # draw_registration_result(source_down, target_down, result_ransac.transformation)
    #
    # result_icp = refine_registration(source_down, target_down, result_ransac,
    #                                  voxel_size)
    # print(result_icp)
    # draw_registration_result(source, target, result_icp.transformation)
    #

