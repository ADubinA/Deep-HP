from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
from pointnet.losses import ConfusionMatrix, CloseNeighborAccuracy
import seaborn as sn
import matplotlib.pyplot as plt
from pointnet.losses import chamfer_distance,chamfer_distance_with_batch

class LightningPointNet(pl.LightningModule):
    def __init__(self, model, ref_pcd):
        super(LightningPointNet, self).__init__()
        self.model = model
        self.lr=0.001
        self.betas = (0.9, 0.999)
        self.accuracy_f = pl.metrics.Accuracy()
        # self.cm = ConfusionMatrix(num_classes=model.num_classes)
        # self.cm.compute_on_step = False
        self.ref_pcd = ref_pcd
        self.stn = STN3DAffine()
    def on_fit_start(self):
        self.ref_pcd = self.ref_pcd.to(self.device)

    def forward(self, x):
        x = x.transpose(2, 1)
        return self.model(x)[0]
    def training_step(self, batch, batch_idx):
        results = self._step_with_loss(batch, batch_idx)
        self.log("train_loss", results["loss"], prog_bar=True, on_step=True, on_epoch=False)
        # self.log("train_accuracy", results["acc"], prog_bar=True, on_step=True, on_epoch=False)
        # self.log("train_neighbor_accuracy", results["nei_acc"], prog_bar=True, on_step=True, on_epoch=False)
        tensorboard = self.logger.experiment
        # tensorboard.add_histogram("value_distribution on train", torch.flatten(results["pred"]), self.global_step)
        pred = results["pred"]
        for idx in range(len(batch["file_name"])):
            target = batch["target"][idx]
            file_name = batch["file_name"][idx]

            tensorboard.add_text(f"results of epoch {self.current_epoch}",
                                 f"results of {pred[idx]} (was really {target}) file name is {file_name}",
                                 self.global_step)
        return results
        # return loss
    def _step_with_loss(self, batch, batch_idx):
        points, target = batch["data"],batch["target"]
        trans = self.stn(points.transpose(2, 1))
        points_z = torch.clone(points)

        ones = torch.ones((points_z.shape[0], points_z.shape[1],1))
        if points_z.is_cuda:
            ones = ones.cuda()
        points_z = torch.cat((points_z,ones), dim = -1)
        # points_z = points_z.transpose(2, 1)
        points_z = torch.bmm(points_z, trans)
        points_z = points_z[:,:,0:3]
        # points_z = points_z.transpose(2, 1)
        # points_z[:,:,2] = points[:,:,2] + pred_z1

        loss = chamfer_distance(points_z,self.ref_pcd)

        return {"loss":loss, "pred_points":points_z, "pred": trans}
    def validation_step(self, batch, batch_idx):
        results = self._step_with_loss(batch, batch_idx)
        pred = results["pred"]
        self.log("validation_loss", results["loss"], prog_bar=True, on_step=False, on_epoch=True)
        # self.log("validation_accuracy", results["acc"], prog_bar=True, on_step=False, on_epoch=True)
        # self.log("validation_neighbor_accuracy", results["nei_acc"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("learning_rate", self.lr, prog_bar=False)
        # self.cm(pred, batch["target"])
        tensorboard = self.logger.experiment
        for idx in range(len(batch["file_name"])):
            target = batch["target"][idx]
            file_name = batch["file_name"][idx]

            tensorboard.add_text(f"results of epoch {self.current_epoch}",
                                     f"results of {pred[idx]} (was really {target}) file name is {file_name}",
                                     self.global_step)
        return results

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=self.betas)
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer)
        return optimizer#, scheduler



class PointNetReg(nn.Module):
    def __init__(self, feature_transform=True):
        super(PointNetReg, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x, trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=True):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x, trans, trans_feat

class STN3DAffine(nn.Module):
    def __init__(self):
        super(STN3DAffine, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 16)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(np.eye(4).astype(np.float32)).view(1,16).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 4, 4)
        # make sure matrix is in form
        x_new = torch.clone(x)

        x_new[:, 0, 0] = torch.pow(x[:, 0, 0],2)
        x_new[:, 1, 1] = torch.pow(x[:, 1, 1],2)
        x_new[:, 2, 2] = torch.pow(x[:, 2, 2],2)

        x_new[:, 3, 0] = 0
        x_new[:, 3, 1] = 0
        x_new[:, 3, 2] = 0
        x_new[:, 3, 3] = 1

        x_new[:, 0, 1] = 0
        x_new[:, 0, 2] = 0
        x_new[:, 1, 0] = 0
        x_new[:, 1, 2] = 0
        x_new[:, 2, 0] = 0
        x_new[:, 2, 1] = 0

        x_new[:, 0, 3] = x[:, 0, 3]
        x_new[:, 1, 3] = x[:, 1, 3]
        x_new[:, 2, 3] = x[:, 2, 3] * 100


        return x_new

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
