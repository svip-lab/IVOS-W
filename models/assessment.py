import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import resnet50
from utils.utils_ipn import ToCudaVariable


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7,
                                 stride=2, padding=3, bias=True)
        self.conv1_p = nn.Conv2d(1, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)
        self.conv1_n = nn.Conv2d(1, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        resnet = resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/16, 1024
        self.res5 = resnet.layer4  # 1/32, 2048

        self.register_buffer('mean', torch.FloatTensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f, in_p, in_g=None):
        f = (in_f - Variable(self.mean)) / Variable(self.std)
        p = torch.unsqueeze(in_p, dim=1).float()  # add channel dim

        if in_g is not None:
            g = torch.unsqueeze(in_g, dim=1).float()  # add channel dim
            x = self.conv1(f) + self.conv1_p(p) + self.conv1_g(g)
        else:
            x = self.conv1(f) + self.conv1_p(p)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 64
        r3 = self.res3(r2)  # 1/8, 128
        r4 = self.res4(r3)  # 1/16, 256
        r5 = self.res5(r4)  # 1/32, 512

        return r5, r4, r3, r2


class AssessNet(nn.Module):
    def __init__(self):
        super(AssessNet, self).__init__()
        self.Encoder = Encoder()  # inputs:: ref: rf, rm / tar: tf, tm

        self.fc1 = nn.Linear(2048, 1)

        self.cnt = 0

    def get_ROI_grid(self, roi, src_size, dst_size, scale=1.):
        # scale height and width
        ry, rx, rh, rw = roi[:, 0], roi[:, 1], scale * \
            roi[:, 2], scale * roi[:, 3]

        # convert ti minmax
        ymin = ry - rh / 2.
        ymax = ry + rh / 2.
        xmin = rx - rw / 2.
        xmax = rx + rw / 2.

        h, w = src_size[0], src_size[1]
        # theta
        theta = ToCudaVariable([torch.zeros(roi.size()[0], 2, 3)])[0]
        theta[:, 0, 0] = (xmax - xmin) / (w - 1)
        theta[:, 0, 2] = (xmin + xmax - (w - 1)) / (w - 1)
        theta[:, 1, 1] = (ymax - ymin) / (h - 1)
        theta[:, 1, 2] = (ymin + ymax - (h - 1)) / (h - 1)

        # inverse of theta
        inv_theta = ToCudaVariable([torch.zeros(roi.size()[0], 2, 3)])[0]
        det = theta[:, 0, 0] * theta[:, 1, 1]
        adj_x = -theta[:, 0, 2] * theta[:, 1, 1]
        adj_y = -theta[:, 0, 0] * theta[:, 1, 2]
        inv_theta[:, 0, 0] = w / (xmax - xmin)
        inv_theta[:, 1, 1] = h / (ymax - ymin)
        inv_theta[:, 0, 2] = adj_x / det
        inv_theta[:, 1, 2] = adj_y / det
        # make affine grid
        fw_grid = F.affine_grid(theta, torch.Size(
            (roi.size()[0], 1, dst_size[0], dst_size[1])), align_corners=True)
        bw_grid = F.affine_grid(inv_theta, torch.Size(
            (roi.size()[0], 1, src_size[0], src_size[1])), align_corners=True)
        return fw_grid, bw_grid, theta

    def all2yxhw(self, mask, scale=1.0):
        np_mask = mask.data.cpu().numpy()

        np_yxhw = np.zeros((np_mask.shape[0], 4), dtype=np.float32)
        for b in range(np_mask.shape[0]):
            mys, mxs = np.where(np_mask[b] >= 0.49)
            all_ys = np.concatenate([mys])
            all_xs = np.concatenate([mxs])

            if all_ys.size == 0 or all_xs.size == 0:
                # if no pixel, return whole
                ymin, ymax = 0, np_mask.shape[1]
                xmin, xmax = 0, np_mask.shape[2]
            else:
                ymin, ymax = np.min(all_ys), np.max(all_ys)
                xmin, xmax = np.min(all_xs), np.max(all_xs)

            # make sure minimum 128 original size
            if (ymax - ymin) < 128:
                res = 128. - (ymax - ymin)
                ymin -= int(res / 2)
                ymax += int(res / 2)

            if (xmax - xmin) < 128:
                res = 128. - (xmax - xmin)
                xmin -= int(res / 2)
                xmax += int(res / 2)

            # apply scale
            # y = (ymax + ymin) / 2.
            # x = (xmax + xmin) / 2.
            orig_h = ymax - ymin + 1
            orig_w = xmax - xmin + 1

            ymin = np.maximum(-5, ymin - (scale - 1) / 2. * orig_h)
            ymax = np.minimum(
                np_mask.shape[1] + 5, ymax + (scale - 1) / 2. * orig_h)
            xmin = np.maximum(-5, xmin - (scale - 1) / 2. * orig_w)
            xmax = np.minimum(
                np_mask.shape[2] + 5, xmax + (scale - 1) / 2. * orig_w)

            # final ywhw
            y = (ymax + ymin) / 2.
            x = (xmax + xmin) / 2.
            h = ymax - ymin + 1
            w = xmax - xmin + 1

            yxhw = np.array([y, x, h, w], dtype=np.float32)

            np_yxhw[b] = yxhw

        return ToCudaVariable([torch.from_numpy(np_yxhw.copy()).float()])[0]

    # def forward(self, tf, tm, tp, tn, gm, loss_weight):  # b,c,h,w // b,4 (y,x,h,w)
    def forward(self, tf, tp):  # b,c,h,w // b,4 (y,x,h,w)
        tm = (tp > 0.5).float()
        tb = self.all2yxhw(tm, scale=1.5)

        oh, ow = tf.size()[2], tf.size()[3]  # original size
        fw_grid, bw_grid, theta = self.get_ROI_grid(
            tb, src_size=(oh, ow), dst_size=(256, 256), scale=1.0)

        #  Sample target frame
        tf_roi = F.grid_sample(tf, fw_grid, align_corners=True)
        tp_roi = F.grid_sample(torch.unsqueeze(tp, dim=1).float(), fw_grid, align_corners=True)[:, 0]

        # run Siamese Encoder
        tr5, tr4, tr3, tr2 = self.Encoder(tf_roi, tp_roi)

        tr5_flat = nn.functional.avg_pool2d(tr5, 8).squeeze()
        quality_pred = self.fc1(tr5_flat)

        return quality_pred
