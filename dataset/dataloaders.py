import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import h5py
import subprocess
import shlex
import json
import glob
import pandas as pd
import open3d as o3d

class VisceralData(Dataset):
	def __init__(
		self,
		data_path,
		labels_path,
		num_points=1024,
		intensity_range=(500, 5000),
		randomize_data=False,
		ignore_list=[]
	):
		super(VisceralData, self).__init__()
		self.ignore_list = ignore_list
		self.labels_csv = self.load_labels(labels_path)
		self.classes = self.labels_csv.groupby('label')
		self.paths = self.load_paths(data_path)
		self.intensity_range = intensity_range
		self.num_points = num_points
		self.randomize_data = randomize_data


	def load_labels(self,labels_path):
		df = pd.read_csv(labels_path)
		df = df[df.label >= 0]
		df = df[~df["label"].isin(self.ignore_list)]
		return df

	def __getitem__(self, idx):
		# if self.randomize_data: current_points = self.randomize(idx)
		file_path = self.paths[idx]
		current_points = self.file_loader(file_path)

		# todo fix duplicates
		label = torch.tensor(int(self.labels_csv.loc[self.labels_csv['file_name'] == self.get_file_name(file_path)].label))
		return {"data":current_points,"target": label,"file_name":os.path.basename(file_path)}

	def __len__(self):
		return len(self.paths)

	@staticmethod
	def get_file_name(file_path):
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
		pcd_tensor = self.preprocess_pcd(pcd_tensor)
		return pcd_tensor

	@staticmethod
	def preprocess_pcd(pcd_tensor):
		pcd_tensor = pcd_tensor - torch.mean(pcd_tensor, dim=1, keepdim=True)
		return pcd_tensor

	def load_paths(self, folder_path):
		paths = []
		for file_path in glob.glob(os.path.join(folder_path, "*.ply")):
			file_name = os.path.basename(file_path)
			if file_name in self.labels_csv["file_name"].values:
				paths.append(file_path)
			else:
				print(f"{file_name} was removed")

		return paths


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