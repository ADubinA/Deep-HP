import glob
import json
import os

import numpy as np
import tqdm

from network.SkeletonHPDataloader import GeometricHPSDataset
import os.path as osp
import torch.nn.functional as F
from datetime import datetime
from network.pointnet2 import Net
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import open3d as o3d
from network.deep_utils import visualize_batch_result, gen_colormap
from tensorboardX import SummaryWriter
from classical_registration.feature_hyper_skeleton import HyperSkeleton
from classical_registration.utils import NumpyEncoder, percent_slicer
from network.deep_utils import RandomCrop,RandomScaleAxis
import math

class DeepHyperSkeleton(HyperSkeleton):
    def __init__(self, path, model, bounds=(np.array([0,0,0]), np.array([1,1,1])),voxel_down_sample_size=2,
                 cluster_func=None, cluster_colors=None, has_labels=True, has_centers=True):
        super().__init__(path, bounds,voxel_down_sample_size, cluster_func, cluster_colors, has_labels, has_centers)
        self.model = model
        self.model_point_size = 8000
        self.g = None
    def create_global_features(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.down_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
        colormap = gen_colormap(self.n_clusters)
        g = self.create_graph(self.down_pcd)
        model_pcd = self.down_pcd.uniform_down_sample(math.ceil(len(self.down_pcd.points)/self.model_point_size))
        model_data = Data(x=torch.tensor(np.asarray(model_pcd.normals),device=device,dtype=torch.float),
                          pos=torch.tensor(np.asarray(model_pcd.points),device=device,dtype=torch.float),
                          batch=torch.zeros(len(model_pcd.points),device=device,dtype=torch.int64))

        model_data = T.NormalizeScale()(model_data)
        results = self.model(model_data).argmax(dim=1).detach().cpu().numpy()
        pcd_tree = o3d.geometry.KDTreeFlann(model_pcd)
        k = 10
        for point_index in g.nodes.keys():
            [_, nei, _] = pcd_tree.search_knn_vector_3d(g.nodes[point_index]["point"], k)
            chosen_label = np.argmax(np.bincount(results[nei]))
            g.nodes[point_index]["label"] = chosen_label
            g.nodes[point_index]["color"] = colormap[chosen_label]
            self.down_pcd.colors[g.nodes[point_index]["index"]] = colormap[chosen_label]


        different_cluster_edges = [edge for edge in g.edges if g.nodes[edge[0]]["label"] != g.nodes[edge[1]]["label"]]
        g.remove_edges_from(different_cluster_edges)
        self.g = g
        self.create_featured_pcd(g)



def train_iter(epoch, model, train_loader,writer,log_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    best_acc = total_loss = correct_nodes = total_nodes = 0
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        loss = F.nll_loss(out, data.y.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes

        writer.add_scalar("train_loss", loss)
        if (i + 1) % 3 == 0:
            acc = correct_nodes / total_nodes
            print(f'[{i+1}/{len(train_loader)}] Loss: {total_loss / 10:.4f} '
                  f'Train Acc: {acc:.4f}')
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), osp.join(log_dir,"best_model.pt"))
                print("best model found and save")

            total_loss = correct_nodes = total_nodes = 0

    visualize_batch_result(data, out.argmax(dim=1), save_path=osp.join(log_dir, str(epoch)))

def val_iter(epoch, model, val_loader,writer,log_dir):
    with torch.no_grad():
        model.eval()
        device = "cpu"#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        best_acc = total_loss = correct_nodes = total_nodes = 0
        for i, data in enumerate(val_loader):
            data = data.to(device)
            out = model(data)

            loss = F.nll_loss(out, data.y.long())
            total_loss += loss.item()
            correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
            total_nodes += data.num_nodes

            writer.add_scalar("validation_loss", loss)
            if (i + 1) % 3 == 0:
                acc = correct_nodes / total_nodes
                print(f'[{i+1}/{len(val_loader)}] validation Loss: {total_loss / 10:.4f} '
                      f'validation Acc: {acc:.4f}')
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), osp.join(log_dir,"val_best_model.pt"))
                    print("best validation model found and save")

                total_loss = correct_nodes = total_nodes = 0

        visualize_batch_result(data, out.argmax(dim=1), save_path=osp.join(log_dir, str(epoch)))


def train():

    log_dir = f'runs/{time_string}'
    comment = "retraining on nmdid"
    train_path = r"D:\datasets\nmdid\labeled"
    val_path =r"D:\datasets\VerSe2020\train_hp_labeled"
    transform = T.Compose([
        RandomCrop(torch.tensor([1, 1, 0.3]), torch.tensor([2.0, 2.0, 2.0])),
        T.FixedPoints(8000),
        # T.GridSampling(0.01),
        T.RandomTranslate(0.05),
        RandomScaleAxis((0.95, 1.05), 2),
        T.RandomRotate(45, axis=0),
        T.RandomRotate(45, axis=1),
        T.RandomRotate(45, axis=2)
    ])
    pre_transform = T.NormalizeScale()
    train_dataset = GeometricHPSDataset(train_path,pre_transform=pre_transform, transform=transform,num_classes=num_classes)
    test_dataset = GeometricHPSDataset(val_path,pre_transform=pre_transform,num_classes=num_classes)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                             num_workers=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(train_dataset.num_classes).to(device)
    writer = SummaryWriter(log_dir,comment=comment)

    for epoch in tqdm.tqdm(range(1, 50)):
        train_iter(epoch, model, train_loader, writer, log_dir)
        # val_iter(epoch, model, test_loader, writer, log_dir)





if __name__ == "__main__":
    time_string = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 64
    train()
