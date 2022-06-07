import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F

class Loss_PIP(nn.Module):
    def __init__(self, lamb, alpha, tau, r, sigma, device):
        super(Loss_PIP, self).__init__()
        self.lamb = lamb
        self.alpha = alpha
        self.tau = tau
        self.r = r
        self.sigma = sigma
        self.device = device

        x_kernel = cv2.getGaussianKernel(2*r+1, sigma).reshape(1, -1)
        y_kernel = cv2.getGaussianKernel(2*r+1, sigma).reshape(-1, 1)
        kernel = x_kernel * y_kernel
        kernel = torch.Tensor(kernel).view(1, 1, 2*r+1, 2*r+1)
        self.conv = torch.nn.Conv2d(1, 1, 2*r+1, bias=False)
        self.conv.weight = torch.nn.Parameter(kernel, requires_grad=False)
        for param in self.conv.parameters():
            param.requires_grad = False
        self.conv = self.conv.to(device)

    def forward(self, logits, bboxes, labels):
        # outputs: B x C x H x W
        # bboxes: B x 56 x 5
        alpha = self.alpha
        num_rc = 1e-5
        loss_rc = torch.zeros([1]).to(self.device)
        logits_soft = logits.softmax(dim=1)
        for i in range(bboxes.shape[0]):
            for j in range(bboxes.shape[1]):
                if bboxes[i, j, -1] == -1:
                    continue
                x_min, y_min, x_max, y_max, cls_id = bboxes[i][j]
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h * w == 0:
                    continue
                loss_rc -= alpha * torch.sum(torch.log(torch.max(logits_soft[i, cls_id, y_min:y_max + 1, x_min:x_max + 1], dim=0).values)) + \
                    (1 - alpha) * torch.sum(torch.log(1 - torch.min(logits_soft[i, cls_id, y_min:y_max + 1, x_min:x_max + 1], dim=0).values)) + \
                    alpha * torch.sum(torch.log(torch.max(logits_soft[i, cls_id, y_min:y_max + 1, x_min:x_max + 1], dim=1).values)) + \
                    (1 - alpha) * torch.sum(torch.log(1 - torch.min(logits_soft[i, cls_id, y_min:y_max + 1, x_min:x_max + 1], dim=1).values))
                num_rc += h + w

        B, _, H, W = logits.shape
        nop = torch.ones((B, H, W), dtype=torch.float, requires_grad=False).to(self.device)
        dis = torch.zeros((B, H, W), dtype=torch.float, requires_grad=False).to(self.device)
        for i in range(bboxes.shape[0]):
            for j in range(bboxes.shape[1]):
                if bboxes[i, j, -1] == -1:
                    continue
                x_min, y_min, x_max, y_max, _ = bboxes[i, j]
                tmp = torch.zeros((H, W), dtype=torch.float).to(self.device)
                tmp[y_min:y_max+1, x_min:x_max+1] = 1
                nop[i] += tmp
                tmp[y_min+1:y_max, x_min+1:x_max] = 0
                dis[i] += tmp

        dis = F.pad(dis.unsqueeze(1), (self.r, self.r, self.r, self.r), mode='reflect')
        dis = self.conv(dis).squeeze() + 1
        Gamma = ((torch.sigmoid(nop * dis / (nop * dis).max()) - 0.5) * self.tau + 1)
        g_map =  Gamma / torch.sum(Gamma, dim=(1, 2), keepdim=True) * H * W
        loss_cls = F.cross_entropy(logits, labels, ignore_index=255, reduce=False)
        return self.lamb * loss_rc / num_rc + (g_map * loss_cls).mean()

def resize_labels(labels, size):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    for label in labels:
        label = label.float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.LongTensor(new_labels)
    return new_labels
    
def resize_labels_bboxes(labels, bboxes, size):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    bboxes = bboxes.float().detach().cpu().numpy()
    for i in range(len(labels)):
        label = labels[i].float().detach().cpu().numpy()
        h, w = label.shape
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
        bboxes[i, :, [0, 2]] = bboxes[i, :, [0, 2]] * size[1] / w
        bboxes[i, :, [1, 3]] = bboxes[i, :, [1, 3]] * size[0] / h
    bboxes[:, :, [0, 2]] = np.clip(bboxes[:, :, [0, 2]], 0, size[1]-1)
    bboxes[:, :, [1, 3]] = np.clip(bboxes[:, :, [1, 3]], 0, size[0]-1)
    bboxes[:, :, :-1] = np.round(bboxes[:, :, :-1])
    new_labels = torch.LongTensor(new_labels)
    bboxes = torch.LongTensor(bboxes)
    return new_labels, bboxes

def build_metrics(model, batch, device):
    pip = Loss_PIP(0.2, 0.5, 5, 5, 2.0, device)
    image_ids, images, labels, bboxes = batch
    logits = model(images.to(device))
    _, _, H, W = logits.shape
    labels, bboxes = resize_labels_bboxes(labels, bboxes, size=(H, W))
    labels, bboxes = labels.to(device), bboxes.to(device)
    preds = torch.argmax(logits, dim=1)
    accuracy = float(torch.eq(preds, labels).sum().cpu()) / (len(image_ids) * logits.shape[2] * logits.shape[3])

    return pip(logits, bboxes, labels), accuracy
                
