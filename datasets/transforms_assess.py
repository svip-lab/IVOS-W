import random
import cv2
import numpy as np
import torch
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

class Resize(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """
    def __init__(self, scales=[0.5, 0.8, 1], size = (854, 480)):
        self.scales = scales
        self.size = size

    def __call__(self, sample):

        img = sample['img']
        label = sample['label']
        prob = sample['prob']

        num_frames = len(img)
        images = []
        probs = []
        labels = []
        for n in range(num_frames):
            images.append(cv2.resize(img[n], self.size, interpolation=cv2.INTER_LINEAR))
            probs.append(cv2.resize(prob[n], self.size, interpolation=cv2.INTER_LINEAR))
            labels.append(cv2.resize(label[n], self.size, interpolation=cv2.INTER_NEAREST))

        sample['img'] = np.stack(images, 0)
        sample['label'] = np.stack(labels, 0)
        sample['prob'] = np.stack(probs, 0)

        return sample


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        img = sample['img']
        label = sample['label']
        prob = sample['prob']

        num_frames = len(img)

        for n in range(num_frames):

            if random.random() < 0.5:

                tmp_img = cv2.flip(img[n], flipCode=1)
                tmp_label = cv2.flip(label[n], flipCode=1)
                tmp_prob = cv2.flip(prob[n], flipCode=1)
                assert tmp_img.ndim == 3
                assert tmp_label.ndim == 2
                assert tmp_prob.ndim == 2
                img[n] = tmp_img
                label[n] = tmp_label
                prob[n] = tmp_prob

        sample['img'] = img
        sample['label'] = label
        sample['prob'] = prob

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):


        img = sample['img']
        label = sample['label']
        prob = sample['prob']

        img = torch.from_numpy(img.transpose((0, 3, 1, 2))).float()
        label = torch.from_numpy(label).float()
        prob = torch.from_numpy(prob).float()

        sample['img'] = img
        sample['label'] = label
        sample['prob'] = prob

        return sample



class RandomAffine(object):

    """
    Affine Transformation to each frame
    """

    def __call__(self, sample):

        img = sample['img']
        label = sample['label']
        prob = sample['prob']

        img_target = np.zeros_like(img)
        label_target = np.zeros_like(label)
        prob_target = np.zeros_like(prob)


        num_objs = len(np.unique(label))

        num_frames = len(img)
        for n in range(num_frames):

            segmap = SegmentationMapsOnImage(label[n], shape=img[n].shape)

            timing = 0
            while True:
                seed = random.randint(0, 2020)
                seq = iaa.Sequential([
                    iaa.Crop(percent=(0.0, 0.1), keep_size=True, seed=seed),
                    iaa.Affine(scale=(0.9, 1.1), shear=(-15, 15), rotate=(-25, 25), seed=seed)
                ])
                img_prob_aug, segmap_aug = seq(image=np.concatenate([img[n], prob[n][:, :, np.newaxis]], 2),
                                               segmentation_maps=segmap)
                img_aug = img_prob_aug[:, :, :3]
                prob_aug = img_prob_aug[:, :, 3]
                if len(np.unique(segmap_aug.get_arr())) == num_objs:
                    img_target[n] = img_aug
                    label_target[n] = segmap_aug.get_arr()
                    prob_target[n] = prob_aug
                    break
                elif timing > 10:
                    img_target[n] = img[n]
                    label_target[n] = label[n]
                    prob_target[n] = prob[n]
                    break
                else:
                    timing += 1

        sample['img'] = img_target
        sample['label'] = label_target
        sample['prob'] = prob_target

        return sample


class AdditiveNoise(object):
    """
    sum additive noise
    """

    def __init__(self, delta=5.0):
        self.delta = delta
        assert delta > 0.0

    def __call__(self, sample):

        img = sample['img']
        num_frames = len(img)
        for n in range(num_frames):
            v = np.random.uniform(-self.delta / 255., self.delta / 255.)
            img[n] += v

            img[n] = np.clip(img[n], 0, 1)
        sample['img'] = img

        return sample


class RandomContrast(object):
    """
    randomly modify the contrast of each frame
    """

    def __init__(self, lower=0.97, upper=1.03):
        self.lower = lower
        self.upper = upper
        assert self.lower <= self.upper
        assert self.lower > 0

    def __call__(self, sample):

        img = sample['img']
        num_frames = len(img)
        for n in range(num_frames):
            v = np.random.uniform(self.lower, self.upper)
            img[n] *= v
            img[n] = np.clip(img[n], 0, 1)
        sample['img'] = img

        return sample


class RandomCrop(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """
    def __init__(self, patch_size=400):
        self.patch_size = patch_size

    def __call__(self, sample):

        img = sample['img']
        label = sample['label']
        prob = sample['prob']

        prob_target = np.zeros_like(prob)
        label_target = np.zeros_like(label)
        img_target = np.zeros_like(img)

        num_frames = len(img)
        for n in range(num_frames):

            size = np.array(img[n].shape[:2])
            assert size.min() > self.patch_size
            timing = 0
            while True:
                h_start = random.randint(0, size[0] - self.patch_size)
                w_start = random.randint(0, size[1] - self.patch_size)

                label_target[n] = label[n][h_start:h_start+self.patch_size, w_start:w_start+self.patch_size]
                indices = np.sort(np.unique(label_target[n]))  # include 0
                if (len(indices) > 1 and 0 in indices) or timing > 10:
                    # re-arrange the indices to ensure the consecutiveness
                    for i, indice in enumerate(indices):
                        label_target[n][label_target[n] == indice] = i
                    prob_target[n] = prob[n][h_start:h_start+self.patch_size, w_start:w_start+self.patch_size]
                    img_target[n] = img[n][h_start:h_start+self.patch_size, w_start:w_start+self.patch_size, :]
                    break
                else:
                    timing += 1

        sample['img'] = img_target
        sample['label'] = label_target
        sample['prob'] = prob_target
        return sample
