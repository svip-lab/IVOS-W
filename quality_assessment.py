import os
import json
import time
import numpy as np
from PIL import Image

from sacred import Experiment
from easydict import EasyDict as edict

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data
from torchvision.transforms import Compose
import datasets.transforms_assess as tr

from models.assessment import AssessNet

from utils.misc import (set_random_seed, save_network_checkpoint, AverageMeter)
from davisinteractive.dataset import Davis
from davisinteractive.metrics import batched_jaccard, batched_f_measure

ex = Experiment()
ex.add_config('./configs/config.yaml')

cudnn.benchmark = False
cudnn.deterministic = True


def sequence_metric(gt_masks, nb_objects, pred_masks, average_over_objects=True, metric_to_optimize='J_AND_F'):
    """ IoU sequence
    :param
        pred_masks: Numpy Array. Array of shape (B x H x W) and type integer giving the
            prediction of the object segmentation.
        sequence: String, the name of the sequence
    :return: float List, IoU value for every frame with pred_masks
    """
    metric = None
    if metric_to_optimize == 'J':
        jaccard = batched_jaccard(
            gt_masks,
            pred_masks,
            average_over_objects=average_over_objects,
            nb_objects=nb_objects)
        metric = jaccard
    elif metric_to_optimize == 'F':
        contour = batched_f_measure(
            gt_masks,
            pred_masks,
            average_over_objects=average_over_objects,
            nb_objects=nb_objects)
        metric = contour
    elif metric_to_optimize == 'J_AND_F':
        jaccard = batched_jaccard(
            gt_masks,
            pred_masks,
            average_over_objects=average_over_objects,
            nb_objects=nb_objects)
        contour = batched_f_measure(
            gt_masks,
            pred_masks,
            average_over_objects=average_over_objects,
            nb_objects=nb_objects)
        metric = .5 * jaccard + .5 * contour

    return metric



class DAVIS2017IoURegression(data.Dataset):
    '''dataloader for agent with bi-dirctional convLSTM
    '''
    def __init__(self,
                 sequences=None,
                 db_root_dir=None,
                 save_result_dir=None,
                 transform=None):

        self.db_root_dir = db_root_dir
        self.save_result_dir = save_result_dir
        self.transform = transform
        self.davis = Davis(davis_root=self.db_root_dir)
        self.sequences = sequences


        self.seqs = []
        with open(os.path.join(self.db_root_dir, 'ImageSets', '2017', 'train.txt')) as f:
            seqs_tmp = f.readlines()
        seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
        self.seqs.extend(seqs_tmp)
        self.seq_list_file = os.path.join(self.db_root_dir, 'ImageSets', '2017', 'train_instances.txt')

        # Precompute the dictionary with the objects per sequence
        if not self._check_preprocess():
            self.seq_dict = self.preprocess(self.db_root_dir, self.seqs, self.seq_list_file)

        sequences = list(self.seq_dict.keys()) if self.sequences is None else self.sequences

        samples_list = []
        for seq in sequences:
            assert seq in self.seq_dict.keys(), f'{seq} not in train set.'

            interaction_list = [int(x.split('-')[-1]) for x in list(np.sort(os.listdir(self.save_result_dir))) if 'interaction' in x]
            scribble_list = [int(x.split('-')[-1]) for x in list(np.sort(os.listdir(os.path.join(self.save_result_dir, f'interaction-{interaction_list[0]}'))))]

            names_list = [x.split('.')[0] for x in list(np.sort(os.listdir(os.path.join(self.db_root_dir, 'JPEGImages/480p/', str(seq)))))]

            img_list = list(map(lambda x: os.path.join('JPEGImages/480p/', str(seq), x+'.jpg'), names_list))
            gt_list = list(map(lambda x: os.path.join('Annotations/480p/', str(seq), x+'.png'), names_list))

            for i in interaction_list:
                for s in scribble_list:
                    for o in self.seq_dict[seq]:

                        prob_list = list(map(lambda x: os.path.join(f'interaction-{i}', f'scribble-{s}', seq, 'probs', f'{o}', x + '.png'), names_list))

                        n_img = len(img_list)
                        for j in range(0, n_img):
                            sample = dict()
                            sample['img_path'] = [img_list[j]]
                            sample['label_path'] = [gt_list[j]]
                            sample['prob_path'] = [prob_list[j]]
                            sample['obj_id'] = o
                            samples_list.append(sample)

        self.samples_list = samples_list


    def _check_preprocess(self):
        if not os.path.isfile(self.seq_list_file):
            return False
        else:
            self.seq_dict = json.load(open(self.seq_list_file, 'r'))
            return True

    def preprocess(self, db_root_dir, seqs, seq_list_file):
        seq_dict = {}
        for seq in seqs:
            # Read object masks and get number of objects
            n_obj = Davis.dataset[seq]['num_objects']

            seq_dict[seq] = list(range(1, n_obj + 1))

        with open(seq_list_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(seqs[0], json.dumps(seq_dict[seqs[0]])))
            for ii in range(1, len(seqs)):
                outfile.write(',\n\t"{:s}": {:s}'.format(seqs[ii], json.dumps(seq_dict[seqs[ii]])))
            outfile.write('\n}\n')

        print('Preprocessing finished')

        return seq_dict


    def __len__(self):
        return len(self.samples_list)


    def __getitem__(self, idx):

        sample = self.samples_list[idx]
        self.samples_list[idx] = None


        img_path = sample['img_path']
        label_path = sample['label_path']
        prob_path = sample['prob_path']
        obj_id = sample['obj_id']

        num_frames = len(img_path)

        images = []
        probs = []
        labels = []

        for n in range(num_frames):
            # img
            img = np.array(Image.open(os.path.join(self.db_root_dir, img_path[n])).convert('RGB'), dtype=np.uint8) / 255.
            img = np.array(img, dtype=np.float32)
            images.append(img)

            # label
            label = Image.open(os.path.join(self.db_root_dir, label_path[n])).convert('P')
            label = np.array(label, dtype=np.uint8)
            label = (label == obj_id).astype(np.uint8)
            labels.append(label)

            # prob
            prob = Image.open(os.path.join(self.save_result_dir, prob_path[n]))
            prob = (np.array(prob) / 255).astype(np.float)
            probs.append(prob)

        # concat
        sample['img'] = np.stack(images, 0)
        sample['label'] = np.stack(labels, 0)
        sample['prob'] = np.stack(probs, 0)

        del sample["img_path"], sample['label_path'], sample['prob_path'], sample['obj_id']

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

@ex.capture
def train(assess_net, optimizer, scheduler, epoch, device, save_result_dir, cfg, metric_to_optimize, _log):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_iou = AverageMeter()
    batch_diff = AverageMeter()

    assess_net.train()

    batch_size = cfg.assess_net.train_batch_size

    train_transform = Compose([
        tr.Resize(size=(854, 480)),
        tr.RandomAffine(),
        tr.AdditiveNoise(),
        tr.RandomContrast(),
        tr.RandomHorizontalFlip(),
        tr.ToTensor(),
    ])
    dataset_train = DAVIS2017IoURegression(transform=train_transform, save_result_dir=save_result_dir,
                                           db_root_dir=cfg.data.root_dir_davis)
    train_loader = data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                   num_workers=cfg.assess_net.num_workers, pin_memory=True, drop_last=True)

    tic = time.time()
    for index, sample in enumerate(train_loader):
        imgs = sample['img']
        probs = sample['prob']
        labels = sample['label']
        sample = None
        masks = (probs > 0.8).float()
        imgs = imgs.to(device)
        probs = probs.to(device)

        N, _, _, H, W = imgs.shape

        # forward pass
        iou_pred = assess_net(imgs.reshape(N, 3, H, W), probs.reshape(N, H, W))

        # calculate target iou
        union = (labels.long() | masks.long()).view(batch_size, -1).sum(1)

        metric_gt = sequence_metric(labels.reshape(N, H, W), 1, masks.reshape(N, H, W),
                                    average_over_objects=False, metric_to_optimize=metric_to_optimize)
        metric_gt = torch.from_numpy(metric_gt).float().to(device)

        loss = 0.
        diff = 0.
        counter = 0
        for n in range(batch_size):
            if union[n] > 0:
                loss += F.mse_loss(iou_pred[n], metric_gt[n])
                diff += (iou_pred[n] - metric_gt[n]).abs().mean()
                counter += 1

        if counter == 0: continue

        loss /= counter
        diff /= counter

        # Backward
        loss.backward()
        for param in assess_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        optimizer.step()

        with torch.no_grad():
            losses.update(loss.item())
            losses_iou.update(loss.item())
            batch_diff.update(diff.item())

        # update time
        batch_time.update(time.time() - tic)
        tic = time.time()

        _log.info(f"Epoch: [{epoch:2d}][{index:3d}/{len(train_loader):3d}]\t"
                  f"Time: {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                  f"Loss: {losses.val:.4f} ({losses.avg:.4f})\t"
                  f"Diff: {diff:.4f} ({batch_diff.avg:.4f})")

    scheduler.step(epoch)

    _log.info(f"* Epoch: [{epoch:3d}]\tLoss: {losses_iou.avg:.6f}\tdiff: {batch_diff.avg:.6f}")



@ex.automain
def main(_run, _log):
    cfg = edict(_run.config)

    # set random seeds
    set_random_seed(2019)

    # Network Builders
    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")

    assess_net = AssessNet()

    assess_net.to(device)

    save_result_dir_train = os.path.join('data', 'quality_assessment')

    # Set up optimizers
    optimizer = torch.optim.SGD(assess_net.parameters(), lr=cfg.assess_net.lr, momentum=cfg.assess_net.momentum,
                                weight_decay=cfg.assess_net.weight_decay)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg.assess_net.gamma)

    metric_to_optimize = cfg.davis_interactive.metric
    print(f"set metric as {metric_to_optimize}")
    # save losses per epoch
    for epoch in range(1, cfg.assess_net.num_epochs + 1):

        _log.info(f"Epoch: {epoch}, current learning rate: {scheduler.get_lr()[0]}")
        train(assess_net, optimizer, scheduler, epoch, device, save_result_dir_train, cfg, metric_to_optimize)

        if epoch % 10 == 0:
            save_network_checkpoint(cfg.ckpt_dir, assess_net)
