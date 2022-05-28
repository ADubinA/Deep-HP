import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import nibabel
from dataset.dicom2pointcloud import load_file, volume_to_pointcloud
import glob
import pandas as pd
import open3d as o3d
import tqdm
import json
class VisceralData(Dataset):
	def __init__(
		self,
		data_path,
		labels_path,
		atlas_path,
		ref_num_points=1024,
		intensity_range=(500, 5000),
		randomize_data=False,
		ref_path=None,
		ignore_list=[]
	):
		super(VisceralData, self).__init__()
		self.atlas_path = atlas_path
		self.labels_path = labels_path
		self.ignore_list = ignore_list
		self.labels = None
		self.intensity_range = intensity_range
		self.ref_num_points = ref_num_points
		self.randomize_data = randomize_data
		self.atlas_axises = []
		self.atlas = {}
		self.num_classes=None
		self._load_labels(labels_path)
		self.paths = self._load_paths(data_path)
		self.ref_pcd =  self._load_reference(ref_path)
		self.load_atlas()
	def _load_reference(self, ref_path):
		if ref_path is None:
			return None
		else:

			ref_pcd = self.file_loader(ref_path)
			indices = torch.randperm(ref_pcd.shape[0])[:self.ref_num_points]
			ref_pcd = ref_pcd[indices,:]
			return ref_pcd[None,:,:]

	def load_atlas(self):
		with open(self.atlas_path, 'r') as f:
			atlas_file = json.load(f)
		self.atlas_axises = []
		self.atlas = {}
		i = 0
		for line, line_group in atlas_file["atlas"].items():
			self.atlas_axises.append(np.array(eval(line)))
			rbfs = []
			for rbf in line_group:
				rbfs.append({
					"mean": np.array(rbf["mean"]),
					"std": np.array(rbf["std"]),
					"axis": np.array(rbf["axis"]),
					"color": np.array(rbf["color"])})
			self.atlas[i] = rbfs
			i += 1

		self.num_classes = atlas_file["num_lines"]
	def _load_labels(self,labels_path):

		with open(self.labels_path, 'r') as f:
			self.labels = json.load(f)["results"]

	def __getitem__(self, idx):
		# if self.randomize_data: current_points = self.randomize(idx)
		file_path = self.paths[idx]
		current_points = self.file_loader(file_path)
		current_points, (low_bound,high_bound) = self.preprocess_pcd(current_points)
		#
		# todo fix duplicates
		try:
			label_dict = [label for label in self.labels if label['path'] == self._get_file_name(file_path)][0]
		except IndexError as e:
			raise IndexError(f"file {file_path} has no label.")

		# set the matches to one hot form
		gaussian_indexes = torch.zeros(self.num_classes,1)
		gaussian_means = torch.zeros(self.num_classes,3)
		for match in label_dict["matches"]:
			match_index = match[0]["index"]
			match = match[1]
			if match["mean"][-1] > high_bound or match["mean"][-1] < low_bound:
				continue
			gaussian_means[match_index,:] = torch.tensor(match["mean"])
			gaussian_indexes[match_index,:] = 1

		label = torch.cat((gaussian_indexes,gaussian_means),dim=-1)
		return {"data":current_points,"target": label,"file_name":os.path.basename(file_path)}

	def __len__(self):
		return len(self.paths)

	@staticmethod
	def _get_file_name(file_path):
		"""
		gets the file name from the file path
		"""
		return os.path.basename(file_path)
	# def randomize(self, idx):
	# 	pt_idxs = np.arange(0, self.num_points)
	# 	np.random.shuffle(pt_idxs)
	# 	return self.data[idx, pt_idxs].copy()

	def file_loader(self, path):
		"""
		from the given file path return the loaded point cloud
		:param path: path to the file
		:type path: str
		:return: the loaded pointcloud, ready for the network
		:rtype: torch.tensor
		"""
		# print(path)
		# volume = dicom2pointcloud.load_file(path)
		# volume = self.volume_preprocess(volume)
		#
		# pcd = dicom2pointcloud.volume_to_pointcloud(volume, 1024, intensity_range=self.intensity_range)
		# return pcd
		pcd = o3d.io.read_point_cloud(path)
		pcd_tensor = torch.from_numpy(np.asarray(pcd.points)).float()
		return pcd_tensor

	@staticmethod
	def preprocess_pcd(pcd):
		# pcd_tensor = pcd_tensor - torch.mean(pcd_tensor, dim=1, keepdim=True)
		num_points = 1024
		low_bound = 200 #np.random.randint(0, torch.max(pcd[:,2])-self.subvolume_slice_size)
		subvolume_slice_size = 400
		high_bound = low_bound+ subvolume_slice_size
		pcd = pcd[pcd[:, 2] < high_bound]
		pcd = pcd[pcd[:, 2] >= low_bound]

		# resample
		indices = torch.randperm(pcd.shape[0])[:num_points]
		pcd = pcd[indices, :]
		pcd = pcd*0.001
		return pcd.view(3,-1), (low_bound, high_bound)

	def _load_paths(self, folder_path):
		paths = []
		for file_path in glob.glob(os.path.join(folder_path, "*.ply")):
			file_name = os.path.basename(file_path)
			file_label = [label for label in self.labels if file_name == label["path"]]
			if len(file_label)==0:
				print(f"{file_name} was removed, as it doesn't have a lable")
			elif file_label[0].get("matches") is None:
				print(f"{file_name} was removed, as the classical algorithm failed registration it.")
			else:
				paths.append(file_path)


		return paths

class NiiData(Dataset):
	def __init__(
			self,
			data_path,
			num_points=1024,
			intensity_range=(500, 5000),
			randomize_data=False,
			ref_path=None,
			load_only_list=None,
			ref_num_points=1024*8,
			subvolume_slice_size=400,
			resamples_per_file=100

	):
		self.data_path = data_path
		self.num_points = num_points
		self.intensity_range = intensity_range
		self.randomize_data = randomize_data
		self.ref_path = ref_path
		self.ref_num_points = ref_num_points
		self.load_only_list = load_only_list
		self.subvolume_slice_size = subvolume_slice_size

		self.paths = self._load_paths(self.data_path)
		self.ref_pcd = self._load_reference()
		self.pcds = self._load_all_ply()

	def __getitem__(self, idx):
		# file_path = self.paths[idx]
		# loaded_dict = self.file_loader(file_path)
		loaded_dict = self.pcds[idx]
		file_name = loaded_dict["file_name"]
		loaded_dict = self.preprocess_pcd(loaded_dict["data"])
		loaded_dict["file_name"] = file_name
		return loaded_dict

	def __len__(self):
		# return len(self.paths)
		return len(self.pcds)

	def _load_reference(self):
		if self.ref_path is None:
			return None
		else:

			ref_pcd = self.file_loader(self.ref_path)["data"]
			indices = torch.randperm(ref_pcd.shape[0])[:self.ref_num_points]
			ref_pcd = ref_pcd[indices,:]
			return ref_pcd[None,:,:]

	def _load_paths(self, folder_path):
		paths = []
		for file_path in glob.glob(os.path.join(folder_path, "*.ply")):
			file_name = os.path.basename(file_path)
			if self.load_only_list is None or file_name in self.load_only_list:
				paths.append(file_path)
			else:
				print(f"{file_name} was removed")
		return paths

	def _load_all_ply(self):
		print("loading pcds")
		pcds = []
		for file_path in tqdm.tqdm(self.paths):
			file_name = os.path.basename(file_path)
			if self.load_only_list is None or file_name in self.load_only_list:
				pcds.append(self.file_loader(file_path))
			else:
				print(f"{file_name} was removed")
		return pcds
	def preprocess_pcd(self, pcd):
		# limit size of the slice

		start_z_index = 200 #np.random.randint(0, torch.max(pcd[:,2])-self.subvolume_slice_size)
		pcd = pcd[pcd[:, 2] < start_z_index+self.subvolume_slice_size]
		pcd = pcd[pcd[:, 2] >= start_z_index]

		# resample
		indices = torch.randperm(pcd.shape[0])[:self.num_points]
		pcd = pcd[indices, :]
		if pcd.shape[0]<self.num_points:
			return None
		# pcd[:, 2] += np.random.randint(50, -50)
		# pcd_tensor = pcd_tensor - torch.mean(pcd_tensor, dim=1, keepdim=True)
		return {"data": pcd, "target": start_z_index}

	def file_loader(self, path):
		"""
		from the given file path return the loaded point cloud
		:param path: path to the file
		:type path: str
		:return: dict with "data" key for the pcd(tensor), and other parameters
		:rtype: dict
		"""
		# if ".nii" in os.path.basename(path):
		# 	volume = load_file(path)
		# 	volume, z_index = self._volume_preprocess(volume)
		#
		# 	pcd = volume_to_pointcloud(volume, self.num_points, intensity_range=self.intensity_range)
		# else:
		pcd = o3d.io.read_point_cloud(path)
		pcd_tensor = torch.from_numpy(np.asarray(pcd.points)).float()
		pcd_dict = {"data": pcd_tensor}
		pcd_dict["file_name"] = os.path.basename(path)[0],
		return pcd_dict

	def _volume_preprocess(self, volume):
		start_z_index = np.random.randint(0, volume.shape[2]-self.subvolume_slice_size)
		volume = volume[:, :, start_z_index:start_z_index+self.subvolume_slice_size]
		return volume, start_z_index
# class RegistrationData(Dataset):
# 	def __init__(self, algorithm, data_class):
# 		super(RegistrationData, self).__init__()
# 		self.algorithm = 'iPCRNet'
#
# 		self.set_class(data_class)
# 		if self.algorithm == 'PCRNet' or self.algorithm == 'iPCRNet':
# 			from .. ops.transform_functions import PCRNetTransform
# 			self.transforms = PCRNetTransform(len(data_class), angle_range=45, translation_range=1)
#
# 	def __len__(self):
# 		return len(self.data_class)
#
# 	def set_class(self, data_class):
# 		self.data_class = data_class
#
# 	def __getitem__(self, index):
# 		template, label = self.data_class[index]
# 		self.transforms.index = index				# for fixed transformations in PCRNet.
# 		source = self.transforms(template)
# 		igt = self.transforms.igt
# 		return template, source, igt
#
#
# class IndentifyData(Dataset):
# 	def __init__(self, algorithm, data_class):
# 		super(IndentifyData, self).__init__()
# 		self.algorithm = 'iPCRNet'
#
# 		self.set_class(data_class)
# 		if self.algorithm == 'PCRNet' or self.algorithm == 'iPCRNet':
# 			from ..ops.transform_functions import PCRNetTransform
# 			self.transforms = PCRNetTransform(len(data_class), angle_range=45, translation_range=1)
#
# 	def __len__(self):
# 		return len(self.data_class)
#
# 	def set_class(self, data_class):
# 		self.data_class = data_class
#
# 	def __getitem__(self, index):
# 		template, label = self.data_class[index]
# 		self.transforms.index = index  # for fixed transformations in PCRNet.
# 		source = self.transforms(template)
# 		return template, source, igt