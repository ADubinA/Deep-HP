import glob
import json
import os

import numpy as np
import tqdm

from SkeletonHPDataloader import GeometricHPSDataset
import os.path as osp
import torch.nn.functional as F
from datetime import datetime
from pointnet2 import Net
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import open3d as o3d
from deep_utils import visualize_single_result, gen_colormap
from tensorboardX import SummaryWriter
from classical_registration.feature_hyper_skeleton import HyperSkeleton
from classical_registration.utils import NumpyEncoder

class DeepHyperSkeleton(HyperSkeleton):
    def __init__(self, path, model, min_z=150, max_z=200, cluster_func=None, cluster_colors=None, has_labels=False, has_centers=False):
        super().__init__(path, min_z, max_z, cluster_func, cluster_colors, has_labels, has_centers)
        self.model = model
        self.model_point_size = 8000
        self.g = None
    def create_global_features(self):
        self.down_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
        colormap = gen_colormap(self.n_clusters)
        g = self.create_graph(self.down_pcd)
        model_pcd = self.down_pcd.uniform_down_sample(int(len(self.down_pcd.points)/self.model_point_size))
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


        different_cluster_edges = [edge for edge in g.edges if g.nodes[edge[0]]["label"] != g.nodes[edge[1]]["label"]]
        g.remove_edges_from(different_cluster_edges)
        self.g = g
        self.create_featured_pcd(g)



def train_iter(epoch, model, train_loader,writer,log_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    visualize_single_result(data,out.argmax(dim=1), save_path=osp.join(log_dir,str(epoch)))

def train():

    log_dir = f'runs/{time_string}'
    comment = "verse first try"
    # path = r"D:\datasets\nmdid\labeled"
    path = r"D:\datasets\VerSe2020\train_hp_labeled"
    transform = T.Compose([
        T.FixedPoints(8000),
        # T.GridSampling(0.01),
        T.RandomTranslate(0.01),
        T.RandomRotate(45, axis=0),
        T.RandomRotate(45, axis=1),
        T.RandomRotate(45, axis=2),

    ])
    pre_transform = T.NormalizeScale()
    train_dataset = GeometricHPSDataset(path,pre_transform=pre_transform, transform=transform,num_classes=num_classes)
    test_dataset = GeometricHPSDataset(path)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                             num_workers=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(train_dataset.num_classes).to(device)
    writer = SummaryWriter(log_dir,comment=comment)

    for epoch in tqdm.tqdm(range(1, 50)):
        train_iter(epoch, model,train_loader,writer,log_dir)


@torch.no_grad()
def test(model, target_path,source_folder_path,save_path, description=""):
    target = DeepHyperSkeleton(target_path,model, min_z=0, max_z=1000, has_labels=True, has_centers=True)
    target.create_global_features()

    subvolume_size = 250
    losses = []
    os.mkdir(os.path.join(save_path))
    for file_path in glob.glob(osp.join(source_folder_path,"*.ply")):
        if "center" in file_path or "label" in file_path or "gl" in file_path:
            continue
        print(file_path)
        for slicer_index in [0,100,200,300,400, 500,600]:

            source = DeepHyperSkeleton(file_path,model,
                                   min_z=slicer_index, max_z=slicer_index+subvolume_size,
                                       has_labels=True, has_centers=True)

            if len(source.base_pcd.points)<100:
                continue

            source.create_global_features()
            transform, loss, fit = source.register(target)
            if source.has_labels:
                segment_losses = source.calculate_label_metric(transform,target,"segment")
            else:
                segment_losses = None
            if source.has_centers:
                center_losses = source.calculate_label_metric(transform, target, "centers")
            else:
                center_losses = None
            losses.append(({"file_name": osp.basename(file_path),
                            "slice_start": slicer_index,
                            "slice_end": slicer_index+subvolume_size,
                            "loss": loss,
                            "transform":transform,
                            "fit_persent": len(fit)/len(source.features),
                            "num_of_features": len(source.features),
                            "segment_losses":segment_losses,
                            "center_losses": center_losses
                            }))

            result_dict = {"results":losses, "description":description,
                           "minimum_cluster_eig": source.minimum_cluster_eig,
                           "minimum_feature_distances": source.minimum_feature_distance,
                           "min_correspondence_percent":source.min_correspondence_percent,
                           "n_clusters":source.n_clusters,
                           "num_bins":source.num_bins,
                           "finish_fit_percent":source.finish_fit_percent,
                           "graph_point_distance":source.graph_point_distance,
                           "max_path_lengths":source.max_path_lengths,
                           "minimum_correspondence_distance":source.minimum_correspondence_distance
                           }
            o3d.io.write_point_cloud(osp.join(save_path,osp.basename(file_path)+str(slicer_index)+".ply"))
            with open(osp.join(save_path,f"{time_string}_results.json"),"w") as f:
                json.dump(result_dict, f, cls=NumpyEncoder)


if __name__ == "__main__":
    time_string = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 64
    # train()

    test_model_path = r"F:\dev\pointermorpher\network\runs\verse02-21-2022_21-40-43\best_model.pt"
    model = Net(num_classes).to(device)
    model.load_state_dict(torch.load(test_model_path))

    test(model, r"D:\datasets\VerSe2020\train\sub-verse823.ply",
                       r"D:\datasets\VerSe2020\*\\",
                       fr"D:\research_results\HyperSkeleton\{time_string}",
                       description="deep HP skeleton test on verse with affine transform. added high rotations to training")
