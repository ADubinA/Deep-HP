from dataset.dicom2pointcloud import *
import pandas as pd
import datetime
import open3d as o3d
import os


def create_folders(folder_path):
    """
    create the folders neccecery for the slice dataset
    :param folder_path: path to output folder
    :return: None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(os.path.join(folder_path,"ply")):
        os.makedirs(os.path.join(folder_path,"ply"))

def create_pointclouds_from_file(file_path, slice_size, intensity_range, sample_num):
    """
    create a list of o3d pointcloud objects, from a given file name
    :param file_path: path to the volume location (nii\gz format)
    :type file_path: str
    :param slice_size: size for the subvolume slice
    :type slice_size: int
    :param intensity_range: range of intensity of the form (min, max) (int)
    :type intensity_range: tuple
    :param sample_num: number of points in the pointcloud to sample
    :type sample_num: int
    :return: a list of o3d pointclouds
    :rtype: list
    """
    volume = load_file(file_path)
    pcd_list = []
    for subvolume_index in range(0, volume.shape[-1], slice_size):
        pcd_list.append((volume_to_pointcloud(volume[:, :, subvolume_index:subvolume_index+slice_size],
                                              sample_num=sample_num, intensity_range=intensity_range), subvolume_index))

    return pcd_list


def create_sliced_pointclouds(volume, intensity_range, sample_num, slice_size, stride):
    """
    create the reference for classification of the other subvolumes.
    :param volume: volume, or path to the volume to be processed
    :type volume: Union[str, np.ndarray]
    :param intensity_range: range of intensity of the form (min, max) (int)
    :type intensity_range: tuple
    :param sample_num: number of points in the pointcloud to sample
    :type sample_num: int
    :param slice_size: size of the slice
    :type slice_size: int
    :param stride: if int how many pixels should the slice stride be.
                   if float will be considered as a fraction of slize size
                   if None, will be considered as the slice_size
    :type stride: Union[int, None, float]
    :return: returns the reference as pcd(o3d), and the subvolume index in the original volume(int)
    :rtype: list
    """
    if type(volume) == str:
        volume = load_file(volume)

    if stride is None:
        stride = slice_size
    elif type(stride) == float:
        stride = int(slice_size*stride)

    # create the list of pointclouds.
    pcd_list = []
    for subvolume_index in range(0, volume.shape[-1], stride):
        pcd_list.append((volume_to_pointcloud(volume[:, :, subvolume_index:subvolume_index+slice_size],
                                              sample_num=sample_num, intensity_range=intensity_range), subvolume_index))

    return pcd_list
def create_defaults(reference_volume, num_slices, stride):
    """
    handles defaults of the create_dataset function
    :param reference_volume: the reference volume
    :type reference_volume: np.ndarray
    :param num_slices: how many slices to make from the volume, if None, will calculate using slice_size.
    :type num_slices: int
    :param stride: if int how many pixels should the slice stride be.
                   if float will be considered as a fraction of slize size
                   if None, will be considered as the slice_size
    :return:
    :rtype:
    """
    # get the slicing size and slice the volume
    ref_z = reference_volume.shape[-1]
    slice_size = int(ref_z/num_slices)

    if stride is None:
        stride = slice_size
    elif type(stride) == float:
        stride = int(slice_size*stride)
    return slice_size, stride


def create_dataset(reference_path, input_folder, output_folder,num_slices, intensity_range=(500,1000), sample_num=1024, stride=None):
    """
    create the slice dataset 1 from visceral. will label the subvolumes
    :param reference_path: location of the reference file
    :type reference_path: str
    :param input_folder: location to the folder visceral dataset
    :type input_folder: str
    :param output_folder: location to the output folder of visceral dataset, will create if not exist.
    :type output_folder:str
    :param num_slices: how many slices to make from the reference
    :type num_slices: int
    :param intensity_range: range of intensity of the form (min, max) (int)
    :type intensity_range: tuple
    :param sample_num: number of points in the pointcloud to sample
    :type sample_num: int
    :param stride: if int how many pixels should the slice stride be.
                   if float will be considered as a fraction of slize size
                   if None, will be considered as the slice_size
    :type stride: Union[int, None, float]
    :return: None
    """
    # create the labeling table
    df = pd.DataFrame({"file_name": [], "subvolume_index": [], "subvolume_size": [],
                       "origin_file": [], "label": [],"time":[]})
    if os.path.isfile(os.path.join(output_folder,"labels.csv")):
        df = pd.read_csv(os.path.join(output_folder,"labels.csv"))

    # create the the folders if not exist
    create_folders(output_folder)
    # load reference and process to a pointcloud
    ref_volume = load_file(reference_path)
    slice_size, stride = create_defaults(ref_volume, num_slices, stride)

    ref_pcds = create_sliced_pointclouds(ref_volume, intensity_range, sample_num, slice_size=slice_size, stride=stride)
    # get all files
    i = 0
    for file_path in tqdm.tqdm(glob.glob(os.path.join(input_folder,"*.nii*"))):
        file_name = os.path.basename(file_path).split(".")[0]

        # if file_name !="113680_CT_Wb" and i==0:
        #     continue
        # else:
        #     i+=1

        # load file random slice it
        complete = 0
        while complete!=10:
            try:
                pcd_list = create_sliced_pointclouds(file_path,intensity_range,sample_num, slice_size=slice_size, stride=stride)
                complete=10
            except:
                if complete<10:
                    complete+=1
                else:
                    row = {"file_name": 0, "subvolume_index": 0, "subvolume_size": 0,
                           "origin_file": file_name, "label": -1, "true_label": -1,
                           "time": datetime.datetime.utcnow()}
                    df = df.append(row, ignore_index=True)
                    print(f"Error with file {file_path}")
                    continue

        for pcd_tuple in pcd_list:
            pcd = pcd_tuple[0]
            if not pcd:
                continue

            subvolume_index = pcd_tuple[1]
            output_filename = file_name + "_" + str(subvolume_index) + ".ply"

            # create a label and write it to the table
            label_index = find_best_z_cpu(pcd, ref_pcds)
            row = {"file_name": output_filename, "subvolume_index": subvolume_index, "subvolume_size": slice_size,
                   "origin_file": file_name, "label": label_index,"true_label": ref_pcds[label_index][1],
                   "time": datetime.datetime.utcnow()}

            # save the date
            o3d.io.write_point_cloud(os.path.join(output_folder,"ply",output_filename), pcd)
            df = df.append(row, ignore_index=True)
            if i%5==0:
                df.to_csv(os.path.join(output_folder,"labels.csv"))
            i+=1
    df.to_csv(os.path.join(output_folder, "labels.csv"))

def create_reference(volume_path, file_name, save_location, intensity_range=(500,1000), sample_num=0):
    volume = load_file(volume_path)
    pcd = volume_to_pointcloud(volume, sample_num=sample_num, intensity_range=intensity_range)
    o3d.io.write_point_cloud(os.path.join(save_location, file_name+".ply"), pcd)

def create_full_ply_dataset(input_folder, output_folder, sample_num=0):
    for file_path in tqdm.tqdm(glob.glob(os.path.join(input_folder,"*.nii*"))):
        file_name = os.path.basename(file_path).split(".")[0]
        if os.path.exists(os.path.join(output_folder,file_name +".ply")):
            continue

        try:
            create_reference(file_path,file_name,output_folder)
        except Exception as e:
            print ("error at")
            print (file_name)


if __name__ == "__main__":
    # reference_path = r"D:\visceral\Anatomy3-trainingset\ct_wb\10000017_1_CT_wb.nii.gz"
    input_folder = r"D:\visceral\visceral2\retrieval-dataset\ct_wb"
    output_folder = r"D:\visceral\full_skeletons"
    create_full_ply_dataset(input_folder, output_folder)
