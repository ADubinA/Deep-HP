from scipy.interpolate import RegularGridInterpolator
import numpy as np
import open3d as o3d
import pydicom
import nibabel as nib
import os, glob
import torch
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import interpolation
import tqdm
import copy

def emd_scipy(pcl1,pcl2):
    d = cdist(pcl1, pcl2,'euclidean')
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / min(pcl1.shape[0], pcl2.shape[0])

def volume_to_pointcloud(volume: np.ndarray, sample_num = 0,color=None, intensity_range=None):
    """
    generates a pointcloud from a the given volume.
    Args:
        volume: numpy array of dim 3. samples will be taken from this.
        sample_num: number of sample to be taken from the volume. If 0, will get all samples
        intensity_range: tuple of the form (low, high) or None. samples will be within that value.
            if None, pointcloud will be from any range.

    Returns: numpy array of the form

    """
    # get all good values
    if not intensity_range:
        intensity_range = (-float("infinity"), float("infinity"))
    locations = np.argwhere((volume > intensity_range[0]) & (volume <= intensity_range[1]))
    colors = volume[np.where((volume > intensity_range[0]) & (volume <= intensity_range[1]))]
    colors = np.tile(colors.reshape(-1,1),(1,3))
    colors += 1025
    colors /= 5000

    if locations.shape[0] <= sample_num:
        print("wanted {} samples, but the filtered image has only {} values".format(sample_num,colors.shape[0]))
        return False
    # get only a few of them
    if not sample_num:
        idx = np.random.randint(locations.shape[0], size=locations.shape[0])
    else:
        idx = np.random.randint(locations.shape[0], size=sample_num)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(locations[idx])
    if color is None:
        pcd.colors = o3d.utility.Vector3dVector(colors[idx])
    else:
        pcd.paint_uniform_color(color)
    # o3d.visualization.draw_geometries([pcd])

    return pcd


def load_file(path, dict_key="arr_0"):
    """
    Load a volume file to memory
    Args:
        path(string):
            path to the file, needs to be of '.nii', '.nii.gz' or '.npz' format only
        dict_key(string):
            optional key for numpy type file
    Returns:
        numpy array of the volume in float 64 format
    """

    if path.endswith(('.nii', '.nii.gz')):
        image = nib.load(path).get_data()

    elif path.endswith('.npz'):
        image = np.load(path)[dict_key]
    elif path.endswith(".dcm"):
        image = pydicom.read_file(path)
        image = image.pixel_array

    else:
        raise OSError("unknown file was loaded")
    return image.astype("float64")

def registering_metric(source, target):
    # reg_p2p = o3d.registration.registration_icp(
    #     source, target, 0.2, np.identity(4),
    #     o3d.registration.TransformationEstimationPointToPoint())
    # source = copy.deepcopy(source)
    # source = source.transform(reg_p2p.transformation)
    #
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(source)
    # vis.add_geometry(target)
    # vis.run()

    return emd_scipy(np.asarray(source.points), np.asarray(target.points))

def convert_folder_volume2pcd(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_path in tqdm.tqdm(glob.glob(os.path.join(input_folder,"*.nii.gz"))):
        file_name = os.path.basename(file_path).split(".")[0]
        output_path = os.path.join(output_folder,file_name+".ply")

        volume = load_file(file_path)
        pcd = volume_to_pointcloud(volume)
        o3d.io.write_point_cloud(output_path,pcd)

def uncompress_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_path in tqdm.tqdm(glob.glob(os.path.join(input_folder,"*.nii.gz"))):
        file_name = os.path.basename(file_path).split(".")[0]
        output_path = os.path.join(output_folder,file_name+".nii")

        volume = nib.load(file_path)
        nib.save(volume, output_path)


def find_best_z_cpu(moving_pcd, ref_pcds):
    """
    returns the best matching index for the given strides nad window
    :param moving_pcd: the pointcloud subvolume
    :type moving_pcd: o3d.pointcloud
    :param ref_pcds: list of pointclouds, will each one for the best match
    :type ref_pcds: list
    :return: best index of the label for the given search
    :rtype: int
    """
    # init scores
    best_z = 0
    best_score = float("infinity")
    # print("current best score is: " + str(best_score))

    # for every stride calculate the subvolume from the reference, as pointcloud
    for ref_pcd_index in range(len(ref_pcds)):
        ref_pcd = ref_pcds[ref_pcd_index]
        #if exist, get the score
        if not ref_pcd[0]:
            continue
        score = registering_metric(moving_pcd, ref_pcd[0])

        # check if it is the best score
        if score < best_score:
            best_score = score
            best_z = ref_pcd_index
        # print(f"score is: {str(score)} with z of {str(ref_pcd[1])} (best is z {str(best_score)} with {str(best_z)}")

    return best_z


# def find_best_z_gpu(moving_pcd, refs):
#     """
#     returns the best matching index for the given strides nad window
#     :param moving_pcd: the pointcloud subvolume
#     :type moving_pcd: o3d.pointcloud
#     :param refs the reference
#     :type refs: np.ndarray
#     :return: best label for the given search
#     :rtype: int
#     """
#     ref_pcd = volume_to_pointcloud(ref[:, :, i:i + window], sample_num, intensity_range=intensity_range)
#
#
    # return best_z
if __name__ == "__main__":
    uncompress_folder(r"D:\visceral\visceral2\retrieval-dataset\CT_Volumes", r"D:\visceral\visceral2\retrieval-dataset\skeleton_pcd")