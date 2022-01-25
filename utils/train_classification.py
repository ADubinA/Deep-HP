from __future__ import print_function
import random
import torch
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import random_split
# from old_pointnet.dataset import ShapeNetDataset, ModelNetDataset
from old_pointnet.models.classifer_with_mean_location import PointNetCls, feature_transform_regularizer, LightningPointNet
import torch.nn.functional as F
from dataset import dataloaders
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
# def load_model40():
#     if opt.dataset_type == 'shapenet':
#         dataset = ShapeNetDataset(
#             root=opt.dataset,
#             classification=True,
#             npoints=opt.num_points)
#
#         test_dataset = ShapeNetDataset(
#             root=opt.dataset,
#             classification=True,
#             split='test',
#             npoints=opt.num_points,
#             data_augmentation=False)
#     elif opt.dataset_type == 'modelnet40':
#         dataset = ModelNetDataset(
#             root=opt.dataset,
#             npoints=opt.num_points,
#             split='trainval')
#
#         test_dataset = ModelNetDataset(
#             root=opt.dataset,
#             split='test',
#             npoints=opt.num_points,
#             data_augmentation=False)
#     else:
#         exit('wrong dataset type')
blue = lambda x: '\033[94m' + x + '\033[0m'


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--batchSize', type=int, default=32, help='input batch size')
    # parser.add_argument(
    #     '--num_points', type=int, default=1024*4, help='input batch size')
    # parser.add_argument(
    #     '--workers', type=int, help='number of data loading workers', default=4)
    # parser.add_argument(
    #     '--nepoch', type=int, default=250, help='number of epochs to train for')
    # parser.add_argument('--outf', type=str, default='runs', help='output folder')
    # parser.add_argument('--cuda', type=bool, default=True, help='if True, will use cuda')
    # parser.add_argument('--model', type=str, default='', help='model path')
    # parser.add_argument('--train_dataset', type=str, default=r'D:\visceral\train_visceral\ply', help="dataset path")
    # parser.add_argument('--train_dataset_labels', type=str, default=r'D:\visceral\train_visceral\labels.csv', help="dataset path")
    # parser.add_argument('--test_dataset', type=str, default=r'D:\visceral\test_visceral\ply', help="dataset path")
    # parser.add_argument('--test_dataset_labels', type=str, default=r'D:\visceral\test_visceral\labels.csv', help="dataset path")
    # parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
    # parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    #
    # # opt = parser.parse_args()
    # time_sign = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # parser.add_argument('--summary_path', type=str, default=os.path.join(opt.outf,"tb",time_sign), help='output folder')
    # parser.add_argument('--checkpoint_path', type=str, default=os.path.join(opt.outf,"checkpoint",time_sign), help='output folder')
    # opt = parser.parse_args()
    # print(opt)
    seed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)
    batch_size = 4
    workers = 4
    ref_path = r"D:\visceral\full_skeletons\102946_CT_Wb.ply"
    train_dataset_path = r"D:\visceral\full_skeletons"
    run_description = "the value distribution if z=200"

    dataset = dataloaders.NiiData(data_path=train_dataset_path,ref_path=ref_path)
    test_size = int(len(dataset)*(0.1))
    train_dataset, val_dataset = random_split(dataset,[len(dataset)-test_size,test_size])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=True
    )
    val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
        drop_last=True)
    # test_dataloader = torch.utils.data.DataLoader(
    #         val_dataset,
    #         batch_size=opt.batchSize,
    #         shuffle=True,
    #         num_workers=int(opt.workers))

    print(f"train size : = {len(train_dataset)} test size = {len(val_dataset)}")
    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        filename='{epoch}-{validation_loss:.2f}',
        save_top_k=3,
    )

    model = LightningPointNet(PointNetCls(k=1,feature_transform=False), dataset.ref_pcd)
    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    trainer = pl.Trainer(max_epochs=90, gpus=1,logger=tb_logger, log_every_n_steps=10,flush_logs_every_n_steps=100,
                         check_val_every_n_epoch=1, weights_summary='full',
                         callbacks=[VisveralCallback(),checkpoint_callback])

    tb_logger.experiment.add_text("Run description", run_description)
    trainer.fit(model=model, train_dataloader=train_dataloader,val_dataloaders=val_dataloader)
    # trainer.test(test_dataloaders=test_dataloader)
    # train(dataloader, eval_dataloader, test_dataloader, classifier, opt, writer)

class VisveralCallback(Callback):
    # def on_sanity_check_end(self, trainer, pl_module):
    #     """
    #     callback to add the reference pointcloud the tensorboard
    #     :param trainer:
    #     :type trainer: pl.Trainer
    #     :param pl_module:
    #     :type pl_module: pl.LightningModule
    #     :return:
    #     :rtype:
    #     """
    #     ref_path = r"D:\visceral\slice_data1\ply"
    #     reference_pointcloud_names = [f"10000005_1_CT_wb_{i}.ply" for i in [0, 72, 144, 216, 288,360,
    #                                                                         432, 504, 576, 648, 720,
    #                                                                         792, 864]]
    #     tensorboard = pl_module.logger.experiment
    #     dataloader = pl_module.val_dataloader()
    #     i=0
    #     for ref_name in reference_pointcloud_names:
    #         pcd = dataloader.dataset.file_loader(os.path.join(ref_path,ref_name)).view(1,-1,3)
    #         tensorboard.add_mesh(f"reference with label {i}, named {ref_name}", pcd)
    #         i+=1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        tensorboard = pl_module.logger.experiment

        # get bad results as text format for tensorboard
        # pred = outputs["pred"]
        # for idx in range(len(batch["file_name"])):
        #     target = batch["target"][idx]
        #     file_name = batch["file_name"][idx]
        #     if pred[idx] != target:
        #         tensorboard.add_text("results", f"bad results of {pred[idx]} (was really {target}) file name is {file_name}",pl_module.global_step)
        # check if label distribution is ok
        tensorboard.add_histogram("label distribution on validation", batch["target"], pl_module.global_step)
        # tensorboard.add_histogram("value_distribution on validation", torch.flatten(outputs["pred"][]), pl_module.global_step)

    # def on_train_start(self, trainer, pl_module):
    #     add the graph
        # tensorboard = pl_module.logger.experiment
        # loader = trainer.train_dataloader
        # batch= next(iter(loader))
        # tensorboard.add_graph(pl_module, batch["data"].to(pl_module.device))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """
        I want to log how my input is doing, with respect to the
        :param trainer:
        :type trainer:
        :param pl_module:
        :type pl_module:
        :param outputs:
        :type outputs:
        :param batch:
        :type batch:
        :param batch_idx:
        :type batch_idx:
        :param dataloader_idx:
        :type dataloader_idx:
        :return:
        :rtype:
        """
        tensorboard = pl_module.logger.experiment
        tensorboard.add_histogram("label distribution on train", batch["target"], pl_module.global_step)


def train(dataloader,eval_dataloader,test_dataloader, classifier, opt, writer):

    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    epoch = 0
    best_epoch_accuracy = 0
    if opt.cuda:
        classifier.cuda()

    for epoch in range(opt.nepoch):
        scheduler.step()
        for i, data in enumerate(dataloader, 0):
            epoch_step(dataloader, classifier, epoch, opt, writer, optimizer)
            
            if i % 10 == 0:
                epoch_accuracy = epoch_step(eval_dataloader, classifier, epoch, opt, writer)
                if epoch_accuracy >= best_epoch_accuracy:
                    best_epoch_accuracy=epoch_accuracy
                    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.checkpoint_path, epoch))

    epoch_step(test_dataloader, classifier, epoch, opt, writer)


def epoch_step(dataloader,classifier,epoch, opt,writer, optimizer=None):
    num_batch = len(dataloader.dataset) / opt.batchSize
    correct = 0
    accuracy =0
    for i, data in enumerate(dataloader, 0):
        if optimizer is not None:
            optimizer.zero_grad()
            classifier = classifier.train()

        points, target = data
        points = points.transpose(2, 1)
        if opt.cuda:
            points, target = points.cuda(), target.cuda()

        pred, trans, trans_feat = classifier(points)
        pred_choice = pred.data.max(1)[1]

        if optimizer is not None:
            correct = pred_choice.eq(target.data).cpu().sum()
            accuracy = correct.item() / float(opt.batchSize)

            loss = F.nll_loss(pred, target)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), accuracy))
            writer.add_scalar('Loss/train', loss.item(), epoch)
            writer.add_scalar('Accuracy/train', accuracy, epoch)
        else:
            correct += pred_choice.eq(target.data).cpu().sum()

    if optimizer is None:
        accuracy = correct.item() / len(dataloader.dataset)
        print(blue('[%d] test accuracy: %f' % (epoch, accuracy)))
        writer.add_scalar('Accuracy/test', accuracy, epoch)
    return accuracy



if __name__ == "__main__":
        main()