import json

import tqdm, glob,zipfile,os,zlib,shutil
import numpy as np
import nibabel as nib
import pydicom
import open3d as o3d
def affine2d(ds):
    F11, F21, F31 =  ds.ImageOrientationPatient[3:]
    F12, F22, F32 =  ds.ImageOrientationPatient[:3]

    dr, dc = ds.PixelSpacing
    Sx, Sy, Sz = ds.ImagePositionPatient

    return np.array(
        [
            [F11 * dr, F12 * dc, 0, Sx],
            [F21 * dr, F22 * dc, 0, Sy],
            [F31 * dr, F32 * dc, ds.SliceThickness, Sz],
            [0, 0, 0, 1]
        ]
    )

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
        nii = nib.load(path)
        image = nii.get_fdata()
        affine = nii.affine
    elif path.endswith('.npz'):
        image = np.load(path)[dict_key]
        affine = np.eye(4)
    elif path.endswith(".dcm"):
        dcm_image = pydicom.dcmread(path)
        image = dcm_image.pixel_array
        affine = affine2d(dcm_image)

    else:
        raise OSError("unknown file was loaded")
    return image.astype("float64"), affine

def volume_to_pointcloud(volume: np.ndarray, sample_num = 0,color=None, intensity_range=None, affine=None):
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
    # colors += intensity_range[0]
    # colors /= intensity_range[1]

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
    if affine is not None:
        pcd.transform(affine)
    # o3d.visualization.draw_geometries([pcd])

    return pcd

def convert_folder_volume2pcd(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_path in tqdm.tqdm(glob.glob(os.path.join(input_folder,"*.nii.gz"))):
        file_name = os.path.basename(file_path).split(".")[0]
        output_path = os.path.join(output_folder,file_name+".ply")
        if os.path.isfile(output_path):
            continue

        volume = load_file(file_path)
        pcd = volume_to_pointcloud(volume, intensity_range=(1500,3000))
        o3d.io.write_point_cloud(output_path,pcd)

def extract_nmdid_zip_to_nii(in_folder,tmp, output_folder):
    for zip_path in tqdm.tqdm(glob.glob(os.path.join(in_folder, "*.zip"))):
        case_name = os.path.basename(zip_path).replace(".zip","")
        try:
            archive = zipfile.ZipFile(zip_path)

            for file in archive.namelist():
                removed_spaces = file.replace(" ", "").replace("_","").replace(".","").lower()
                if ((("bone" in removed_spaces) and ("3x3" in removed_spaces)) or ("lung" in removed_spaces)) and removed_spaces.endswith("dcm"):
                    archive.extract(file, tmp)
            archive.close()
        except zlib.error:
            print(f"bad zip file {case_name}")
            try:
                shutil.rmtree(os.path.join(tmp, "omi"))
                print("deleted tmp for bad file")
            except:
                print("no needed to delete bad case")
            continue
        for folder in glob.glob(os.path.join(tmp, "*", "*", case_name, "*", "*\\")):
            x = []
            for file in glob.glob(os.path.join(folder, "*.dcm")):
                dcm_image = pydicom.dcmread(file)
                x.append(dcm_image.pixel_array)

            np_image = np.transpose(x, (1, 2, 0))
            path = os.path.normpath(folder)
            study_type = path.split(os.sep)[-1]
            output_path = os.path.join(output_folder, case_name+"_"+study_type + ".nii.gz")

            img = nib.Nifti1Image(np_image, affine2d(dcm_image))  # Save axis for data (just identity)
            img.header.get_xyzt_units()
            img.to_filename(output_path)  # Save as NiBabel file

        shutil.rmtree(os.path.join(tmp,"omi"))
        os.remove(zip_path)



def verse2020_setup(folder_path, output_path):
    for file_path in tqdm.tqdm(glob.glob(os.path.join(folder_path,"rawdata","*\\"))):

        # get names
        file_name = os.path.normpath(file_path).split(os.sep)[-1]
        raw_file_path = glob.glob(os.path.join(folder_path, "rawdata", file_name, "*nii*"))[0]
        seg_file_path = glob.glob(os.path.join(folder_path, "derivatives", file_name, "*nii*"))[0]
        center_file_path = glob.glob(os.path.join(folder_path, "derivatives", file_name, "*json"))[0]

        if os.path.exists(os.path.join(output_path, file_name+".ply")):
            continue
            pass
        print(file_name)
        # handle the data
        if "767" in file_name or "503" in file_name or "507" in file_name:
            intensity = (1000, 10000)
        else:
            intensity = (450, 1000)
        verse2020_raw(raw_file_path, os.path.join(output_path, file_name+".ply"), intensity_range=intensity)
        affine = verse2020_segment(seg_file_path, os.path.join(output_path, file_name+"_labels.ply"))
        verse2020_centers(center_file_path, os.path.join(output_path, file_name+"_centers.ply"),affine=affine)

def verse2020_raw(file_path, save_path, intensity_range=(0,100000)):
    volume, affine = load_file(file_path)
    pcd = volume_to_pointcloud(volume,intensity_range=intensity_range, affine = affine)
    pcd.transform(np.array([[0,-1,0,0],[1,0,0,0],[0,0,0.25,0],[0,0,0,1]]))
    pcd.paint_uniform_color([0.5,0.5,0.5])
    o3d.io.write_point_cloud(save_path, pcd)


def verse2020_segment(file_path, save_path):
    volume, affine = load_file(file_path)
    pcd = volume_to_pointcloud(volume,intensity_range=(0,100), affine = affine)
    pcd.transform(np.array([[0,-1,0,0],[1,0,0,0],[0,0,0.25,0],[0,0,0,1]]))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors) / 40)
    o3d.io.write_point_cloud(save_path, pcd)
    return affine

def verse2020_centers(file_path, save_path, affine=None):
    with open(file_path,"r") as f:
        points_list = json.load(f)
    pcd = o3d.geometry.PointCloud()
    for data_dict in points_list:
        if not data_dict.get("label",False):
            continue

        pcd.points.append([data_dict["X"], data_dict["Y"], data_dict["Z"]])
        pcd.colors.append([data_dict["label"], data_dict["label"], data_dict["label"]])

    pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors) / 40)
    if affine is not None:
        pcd.transform(affine)
    pcd.transform(np.array([[0,-1,0,0],[1,0,0,0],[0,0,0.25,0],[0,0,0,1]]))
    o3d.io.write_point_cloud(save_path, pcd)


def ctpel_setup(folder_path, output_path):

    for zip_path in tqdm.tqdm(glob.glob(os.path.join(folder_path,".zip"))):
        file_name = os.path.normpath(zip_path).split(os.sep)[-1]
        temp_folder = os.path.join(folder_path, file_name+"temp")
        archive = zipfile.ZipFile(zip_path)
        archive.extractall(temp_folder)
        archive.close()

        temp_folder
        # get names
        # raw_file_path = glob.glob(os.path.join(folder_path, "rawdata", file_name, "*nii*"))[0]
        # seg_file_path = glob.glob(os.path.join(folder_path, "derivatives", file_name, "*nii*"))[0]
        # center_file_path = glob.glob(os.path.join(folder_path, "derivatives", file_name, "*json"))[0]

        # handle the data
        # verse2020_raw(raw_file_path, os.path.join(output_path, file_name+".ply"), intensity_range=(500,10000))
        # verse2020_segment(seg_file_path, os.path.join(output_path, file_name+"_labels.ply"))
        # verse2020_centers(center_file_path, os.path.join(output_path, file_name+"_centers.ply"))


def calculate_bone_percent(path, intensity_range=(450, 1000)):
    data = load_file(path)[0]
    pcd = volume_to_pointcloud(data, intensity_range=intensity_range)
    return len(pcd.points)/(data.shape[0]*data.shape[1]*data.shape[2])


if __name__=="__main__":
    verse_path =  r"D:\datasets\VerSe2020\raw_train\\"
    output_path = r"D:\datasets\VerSe2020\new_train\\"
    verse2020_setup(verse_path, output_path)

    verse_path =  r"D:\datasets\VerSe2020\raw_validation\\"
    output_path = r"D:\datasets\VerSe2020\new_validation\\"
    verse2020_setup(verse_path, output_path)