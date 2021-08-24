from __future__ import print_function
# import torch
# import torch.nn.parallel
# import torch.utils.data
# import pytorch_lightning as pl
# import torch.optim as optim
# from models import pointnet2_cls_msg
# from pointnet.losses import chamfer_distance,chamfer_distance_with_batch
# class LightningPointNet(pl.LightningModule):
#     def __init__(self, model, ref_pcd):
#         super(LightningPointNet, self).__init__()
#         self.model = model
#         self.lr=0.001
#         self.betas = (0.9, 0.999)
#         self.accuracy_f = pl.metrics.Accuracy()
#         # self.cm = ConfusionMatrix(num_classes=model.num_classes)
#         # self.cm.compute_on_step = False
#         self.ref_pcd = ref_pcd
#         self.stn = STN3DAffine()
#     def on_fit_start(self):
#         self.ref_pcd = self.ref_pcd.to(self.device)
#
#     def forward(self, x):
#         x = x.transpose(2, 1)
#         return self.model(x)[0]
#     def training_step(self, batch, batch_idx):
#         results = self._step_with_loss(batch, batch_idx)
#         self.log("train_loss", results["loss"], prog_bar=True, on_step=True, on_epoch=False)
#         # self.log("train_accuracy", results["acc"], prog_bar=True, on_step=True, on_epoch=False)
#         # self.log("train_neighbor_accuracy", results["nei_acc"], prog_bar=True, on_step=True, on_epoch=False)
#         tensorboard = self.logger.experiment
#         # tensorboard.add_histogram("value_distribution on train", torch.flatten(results["pred"]), self.global_step)
#         pred = results["pred"]
#         for idx in range(len(batch["file_name"])):
#             target = batch["target"][idx]
#             file_name = batch["file_name"][idx]
#
#             tensorboard.add_text(f"results of epoch {self.current_epoch}",
#                                  f"results of {pred[idx]} (was really {target}) file name is {file_name}",
#                                  self.global_step)
#         return results
#         # return loss
#     def _step_with_loss(self, batch, batch_idx):
#         points, target = batch["data"],batch["target"]
#         trans = self.stn(points.transpose(2, 1))
#         points_z = torch.clone(points)
#
#         ones = torch.ones((points_z.shape[0], points_z.shape[1],1))
#         if points_z.is_cuda:
#             ones = ones.cuda()
#         points_z = torch.cat((points_z,ones), dim = -1)
#         # points_z = points_z.transpose(2, 1)
#         points_z = torch.bmm(points_z, trans)
#         points_z = points_z[:,:,0:3]
#         # points_z = points_z.transpose(2, 1)
#         # points_z[:,:,2] = points[:,:,2] + pred_z1
#
#         loss = chamfer_distance(points_z,self.ref_pcd)
#
#         return {"loss":loss, "pred_points":points_z, "pred": trans}
#     def validation_step(self, batch, batch_idx):
#         results = self._step_with_loss(batch, batch_idx)
#         pred = results["pred"]
#         self.log("validation_loss", results["loss"], prog_bar=True, on_step=False, on_epoch=True)
#         # self.log("validation_accuracy", results["acc"], prog_bar=True, on_step=False, on_epoch=True)
#         # self.log("validation_neighbor_accuracy", results["nei_acc"], prog_bar=True, on_step=False, on_epoch=True)
#         self.log("learning_rate", self.lr, prog_bar=False)
#         # self.cm(pred, batch["target"])
#         tensorboard = self.logger.experiment
#         for idx in range(len(batch["file_name"])):
#             target = batch["target"][idx]
#             file_name = batch["file_name"][idx]
#
#             tensorboard.add_text(f"results of epoch {self.current_epoch}",
#                                      f"results of {pred[idx]} (was really {target}) file name is {file_name}",
#                                      self.global_step)
#         return results
#
#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=self.betas)
#         # scheduler = optim.lr_scheduler.LambdaLR(optimizer)
#         return optimizer#, scheduler



import torch.nn as nn
import torch.nn.functional as F
from pointnet.models.base.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class get_model(nn.Module):
    """
    pointnet++ for calculating the probability of a class, and it's location
    will have 4 times the output of the number of classes. one for probability and 3 for location.
    """
    def __init__(self,num_class,normal_channel=False):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.num_class = num_class
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 4*num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = x.view(B,-1,4)
        x[:,:1 ] = F.log_softmax(x[:,1 ], -1)
        return x,l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.c = 1
    def forward(self, pred, target, trans_feat=None):
        total_loss = F.nll_loss(pred[:,0], target[:,0])
        total_loss += self.c * self.weighted_mse_loss(pred[:,1:], target[:,1:],pred[:,0])
        return total_loss

    @staticmethod
    def weighted_mse_loss(pred, target, weights):
        out = (pred - target) ** 2
        out = out * weights.expand_as(out)
        loss = out.sum(0)  # or sum over whatever dimensions
        return loss
