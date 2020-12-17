from pytorch_lightning.metrics import Metric
import torch
import pytorch_lightning.metrics.functional as F
import numpy as np

class CloseNeighborAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        combine = torch.stack((preds == target, preds == target + 1, preds == target - 1))
        combine = combine.any(dim=0)
        self.correct += torch.sum(combine)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total


class ConfusionMatrix(Metric):
    def __init__(self, num_classes=None, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.add_state("labels", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("predicted", default=torch.tensor([]), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.labels = torch.cat((self.labels, target))
        self.predicted = torch.cat((self.predicted, preds))

    def compute(self):
        return F.confusion_matrix(self.predicted, self.labels, num_classes=self.num_classes)


def chamfer_distance(a, b, method="mean"):
    """
	a: (b, p, 3)
	b: (b, q, 3)
	"""
    diff = a[:, :, None, :] - b[:, None, :, :]  # (b, p, q, 3)
    dist = diff.norm(p=2, dim=3)
    d_min, _ = dist.min(2)
    if method == "mean":
        ch_dist = d_min.mean()
    else:
        ch_dist = d_min.max()
    return ch_dist
def array2samples_distance(array1, array2):
    """
    arguments:
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
    """
    num_point, num_features = array1.shape
    expanded_array1 = np.tile(array1, (num_point, 1))
    expanded_array2 = np.reshape(
            np.tile(np.expand_dims(array2, 1),
                    (1, num_point, 1)),
            (-1, num_features))
    distances = np.linalg.norm(expanded_array1-expanded_array2, axis=1)
    distances = np.reshape(distances, (num_point, num_point))
    distances = np.min(distances, axis=1)
    distances = np.mean(distances)
    return distances

def chamfer_distance_numpy(array1, array2):
    # batch_size, num_point, num_features = array1.shape
    # dist = 0
    return array2samples_distance(array1[0], array2[0])
    # for i in range(batch_size):
    #     av_dist1 = array2samples_distance(array1[i], array2[i])
    #     av_dist2 = array2samples_distance(array2[i], array1[i])
    #     dist = dist + (av_dist1+av_dist2)/batch_size
    # return dist