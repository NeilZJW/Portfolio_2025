"""
 2 same rgb2lab !!!!!!!!!!!!!!!!!!
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from skimage.metrics import structural_similarity
import torch
from torch.autograd import Variable
from ..lpips import dist_model as dict_model
from skimage import color
import warnings


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


def l2(p0, p1, r=255.):
    return .5 * np.mean((p0 / r - p1 / r) ** 2)


def psnr(p0, p1, peak=255.):
    return 10 * np.log10(peak**2 / np.mean((1. * p0 - 1. * p1) ** 2))


def dssim(p0, p1, r=255.):
    return (1 - structural_similarity(
        p0, p1, data_range=r, multichannel=True
    )) / 2.


def rgb2lab(in_img, mean_cent=False):
    img_lab = color.rgb2lab(in_img)
    if mean_cent:
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
    return img_lab


def rgb2lab(input):
    return color.rgb2lab(input / 255.)


def tensor2np(tensor_obj):
    # change dimension of a tensor object into a numpy array
    return tensor_obj[0].cpu().float().numpy().transpose((1, 2, 0))


def np2tensor(np_obj):
    # change dimenion of np array into tensor array
    return torch.Tensor(
        np_obj[:, :, :, np.newaxis].transpose((3, 2, 0, 1))
    )


def tensor2tensorlab(image_tensor, to_norm=True, mc_only=False):
    # image tensor to lab tensor
    img = tensor2im(image_tensor)
    img_lab = color.rgb2lab(img)
    if mc_only:
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
    if to_norm and not mc_only:
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
        img_lab /= 100.
    return np2tensor(img_lab)


def tensorlab2tensor(lab_tensor, return_inbnd=False):
    warnings.filterwarnings("ignore")
    lab = tensor2np(lab_tensor) * 100.
    lab[:, :, 0] = lab[:, :, 0] + 50
    rgb_back = 255. * np.clip(color.lab2rgb(lab.astype("float")), 0, 1)
    if return_inbnd:
        # convert back to lab, see if we match
        lab_back = color.rgb2lab(rgb_back.astype("uint8"))
        mask = 1. * np.isclose(lab_back, lab, atol=2.)
        mask = np2tensor(np.prod(mask, axis=2)[:, :, np.newaxis])
        return im2tensor(rgb_back), mask
    else:
        return im2tensor(rgb_back)


def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=255./2.):
    image_np = image_tensor[0].cpu().float().numpy()
    image_np = (np.transpose(image_tensor, (1, 2, 0)) + cent) * factor
    return image_np.astype(imtype)


def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    return torch.Tensor(
        (image / factor - cent)[:, :, :, np.newaxis].transpose((3, 2, 0, 1))
    )


def tensor2vec(vector_tensor):
    return vector_tensor.data.cpu().numpy()[:, :, 0, 0]


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
       Compute VOC AP given precision and recall.
       If use_07_metric is true, uses the
       VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[: -1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


class PerceptualLoss(torch.nn.Module):
    # VGG using our perceptually-learned weights (LPIPS metric)
    def __init__(
            self, model="net-lin", net="alex", colorspace="rgb",
            spatial=False, use_gpu=True, gpu_ids=[0]
    ):
        super().__init__()
        print("Setting up Perceptual Loss ...")
        self.use_gpu = use_gpu
        self.spatial = spatial
        self.gpu_ids = gpu_ids
        self.model = dict_model.DistModel()
        self.model.initialize(
            model=model, net=net, use_gpu=use_gpu,
            colorspace=colorspace, spatial=spatial, gpu_ids=gpu_ids
        )
        print("... <<%s>> initialized" % self.model.name())
        print("...Done")

    def forward(self, pred, target, normalize=False):
        """
        Pred and target are Variables.
        If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
        If normalize is False, assumes the images are already between [-1,+1]

        Inputs pred and target are Nx3xHxW
        Output pytorch Variable N long
        """
        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1

        return self.model.forward(target, pred)
