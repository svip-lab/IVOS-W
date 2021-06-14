import json
import os

import cv2
import numpy as np
import torch
from libs import utils
from PIL import Image


class DAVIS2017(torch.utils.data.Dataset):
    """DAVIS 2017 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self,
                 split='val',
                 subseq=None,
                 root='',
                 num_frames=None,
                 custom_frames=None,
                 transform=None,
                 retname=False,
                 seq_name=None,
                 obj_id=None,
                 gt_only_first_frame=False,
                 no_gt=False,
                 batch_gt=False,
                 rgb=False,
                 effective_batch=None,
                 prev_round_masks=None,  # f,h,w
                 ):
        """Loads image to label pairs for tool pose estimation
        split: Split or list of splits of the dataset
        root: dataset directory with subfolders "JPEGImages" and "Annotations"
        num_frames: Select number of frames of the sequence (None for all frames)
        custom_frames: List or Tuple with the number of the frames to include
        transform: Data transformations
        retname: Retrieve meta data in the sample key 'meta'
        seq_name: Use a specific sequence
        obj_id: Use a specific object of a sequence (If None and sequence is specified, the batch_gt is True)
        gt_only_first_frame: Provide the GT only in the first frame
        no_gt: No GT is provided
        batch_gt: For every frame sequence batch all the different objects gt
        rgb: Use RGB channel order in the image
        """
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split
        self.subseq = subseq
        self.db_root_dir = root
        self.transform = transform
        self.seq_name = seq_name
        self.obj_id = obj_id
        self.num_frames = num_frames
        self.custom_frames = custom_frames
        self.retname = retname
        self.rgb = rgb
        if seq_name is not None and obj_id is None:
            batch_gt = True
        self.batch_gt = batch_gt
        self.all_seqs_list = []

        self.seqs = []
        for splt in self.split:
            with open(os.path.join(self.db_root_dir, 'ImageSets', '2017', splt + '.txt')) as f:
                seqs_tmp = f.readlines()
            seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
            self.seqs.extend(seqs_tmp)
        self.seq_list_file = os.path.join(self.db_root_dir, 'ImageSets', '2017',
                                          '_'.join(self.split) + '_instances.txt')
        # Precompute the dictionary with the objects per sequence
        if not self._check_preprocess():
            self._preprocess()

        if self.seq_name is None:
            img_list = []
            labels = []
            prevmask_list = []
            for seq in self.seqs:
                images = np.sort(os.listdir(os.path.join(
                    self.db_root_dir, 'JPEGImages/480p/', seq.strip())))
                lab = np.sort(os.listdir(os.path.join(
                    self.db_root_dir, 'Annotations/480p/', seq.strip())))

                if self.subseq is not None:
                    images = images[self.subseq]
                    lab = lab[self.subseq]

                images_path = list(map(lambda x: os.path.join(
                    'JPEGImages/480p/', seq.strip(), x), images))
                lab_path = list(map(lambda x: os.path.join(
                    'Annotations/480p/', seq.strip(), x), lab))
                if num_frames is not None:
                    seq_len = len(images_path)
                    num_frames = min(num_frames, seq_len)
                    frame_vector = np.arange(num_frames)
                    frames_ids = list(
                        np.round(frame_vector*seq_len/float(num_frames)).astype(np.int))
                    frames_ids[-1] = min(frames_ids[-1], seq_len)
                    images_path = [images_path[x] for x in frames_ids]
                    if no_gt:
                        lab_path = [None] * len(images_path)
                    else:
                        lab_path = [lab_path[x] for x in frames_ids]
                elif isinstance(custom_frames, tuple) or isinstance(custom_frames, list):
                    assert min(custom_frames) >= 0 and max(
                        custom_frames) <= len(images_path)
                    images_path = [images_path[x] for x in custom_frames]
                    prevmask_list = [prev_round_masks[x]
                                     for x in custom_frames]
                    if no_gt:
                        lab_path = [None] * len(images_path)
                    else:
                        lab_path = [lab_path[x] for x in custom_frames]
                if gt_only_first_frame:
                    lab_path = [lab_path[0]]
                    lab_path.extend([None] * (len(images_path) - 1))
                elif no_gt:
                    lab_path = [None] * len(images_path)
                if self.batch_gt:
                    obj = self.seq_dict[seq]
                    if -1 in obj:
                        obj.remove(-1)
                    for ii in range(len(img_list), len(images_path)+len(img_list)):
                        self.all_seqs_list.append([obj, ii])
                else:
                    for obj in self.seq_dict[seq]:
                        if obj != -1:
                            for ii in range(len(img_list), len(images_path)+len(img_list)):
                                self.all_seqs_list.append([obj, ii])

                img_list.extend(images_path)
                labels.extend(lab_path)
        else:
            # Initialize the per sequence images for online training
            assert self.seq_name in self.seq_dict.keys(), '{} not in {} set.'.format(
                self.seq_name, '_'.join(self.split))
            names_img = np.sort(os.listdir(os.path.join(
                self.db_root_dir, 'JPEGImages/480p/', str(seq_name))))
            name_label = np.sort(os.listdir(os.path.join(
                self.db_root_dir, 'Annotations/480p/', str(seq_name))))

            if self.subseq is not None:
                names_img = names_img[self.subseq]
                name_label = name_label[self.subseq]

            img_list = list(map(lambda x: os.path.join(
                'JPEGImages/480p/', str(seq_name), x), names_img))
            labels = list(map(lambda x: os.path.join(
                'Annotations/480p/', str(seq_name), x), name_label))
            prevmask_list = []
            if num_frames is not None:
                seq_len = len(img_list)
                num_frames = min(num_frames, seq_len)
                frame_vector = np.arange(num_frames)
                frames_ids = list(
                    np.round(frame_vector * seq_len / float(num_frames)).astype(np.int))
                frames_ids[-1] = min(frames_ids[-1], seq_len)
                img_list = [img_list[x] for x in frames_ids]
                if no_gt:
                    labels = [None] * len(img_list)
                else:
                    labels = [labels[x] for x in frames_ids]
            elif isinstance(custom_frames, tuple) or isinstance(custom_frames, list):
                assert min(custom_frames) >= 0 and max(
                    custom_frames) <= len(img_list)
                img_list = [img_list[x] for x in custom_frames]
                prevmask_list = [prev_round_masks[x] for x in custom_frames]
                if no_gt:
                    labels = [None] * len(img_list)
                else:
                    labels = [labels[x] for x in custom_frames]
            if gt_only_first_frame:
                labels = [labels[0]]
                labels.extend([None]*(len(img_list)-1))
            elif no_gt:
                labels = [None] * len(img_list)
            if obj_id is not None:
                assert obj_id in self.seq_dict[self.seq_name], \
                    "{} doesn't have this object id {}.".format(
                        self.seq_name, str(obj_id))
            if self.batch_gt:
                self.obj_id = self.seq_dict[self.seq_name]
                if -1 in self.obj_id:
                    self.obj_id.remove(-1)
                self.obj_id = [0]+self.obj_id

        assert (len(labels) == len(img_list))

        if effective_batch:
            self.img_list = img_list * effective_batch
            self.labels = labels * effective_batch
        else:
            self.img_list = img_list
            self.labels = labels
            self.prevmasks_list = prevmask_list

        # print('Done initializing DAVIS2017 '+'_'.join(self.split)+' Dataset')
        # print('Number of images: {}'.format(len(self.img_list)))
        # if self.seq_name is None:
        #     print('Number of elements {}'.format(len(self.all_seqs_list)))

    def _check_preprocess(self):
        _seq_list_file = self.seq_list_file
        if not os.path.isfile(_seq_list_file):
            return False
        else:
            self.seq_dict = json.load(open(self.seq_list_file, 'r'))
            return True

    def _preprocess(self):
        self.seq_dict = {}
        for seq in self.seqs:
            # Read object masks and get number of objects
            name_label = np.sort(os.listdir(os.path.join(
                self.db_root_dir, 'Annotations/480p/', seq)))
            label_path = os.path.join(
                self.db_root_dir, 'Annotations/480p/', seq, name_label[0])
            _mask = np.array(Image.open(label_path))
            _mask_ids = np.unique(_mask)
            n_obj = _mask_ids[-1]

            self.seq_dict[seq] = list(range(1, n_obj+1))

        with open(self.seq_list_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(
                self.seqs[0], json.dumps(self.seq_dict[self.seqs[0]])))
            for ii in range(1, len(self.seqs)):
                outfile.write(',\n\t"{:s}": {:s}'.format(
                    self.seqs[ii], json.dumps(self.seq_dict[self.seqs[ii]])))
            outfile.write('\n}\n')

        print('Preprocessing finished')

    def __len__(self):
        if self.seq_name is None:
            return len(self.all_seqs_list)
        else:
            return len(self.img_list)

    def __getitem__(self, idx):
        img, gt, prev_round_mask = self.make_img_gt_mask_pair(idx)

        pad_img, pad_info = utils.apply_pad(img)
        pad_gt = utils.apply_pad(gt, padinfo=pad_info)  # h,w,n
        sample = {'image': pad_img, 'ori': img, 'gt': pad_gt}

        if self.retname:
            if self.seq_name is None:
                obj_id = self.all_seqs_list[idx][0]
                img_path = self.img_list[self.all_seqs_list[idx][1]]
            else:
                obj_id = self.obj_id
                img_path = self.img_list[idx]
            seq_name = img_path.split('/')[-2]
            frame_id = img_path.split('/')[-1].split('.')[-2]
            if self.subseq is not None:
                frame_id = f'{self.subseq.index(int(frame_id))}'.zfill(5)
            sample['meta'] = {'seq_name': seq_name,
                              'frame_id': frame_id,
                              'obj_id': obj_id,
                              'im_size': (img.shape[0], img.shape[1]),
                              'pad_size': (pad_img.shape[0], pad_img.shape[1]),
                              'pad_info': pad_info}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_img_gt_mask_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        prev_round_mask_tmp = self.prevmasks_list[idx]
        if self.seq_name is None:
            obj_id = self.all_seqs_list[idx][0]
            img_path = self.img_list[self.all_seqs_list[idx][1]]
            label_path = self.labels[self.all_seqs_list[idx][1]]
        else:
            obj_id = self.obj_id
            img_path = self.img_list[idx]
            label_path = self.labels[idx]
        seq_name = img_path.split('/')[-2]
        n_obj = 1 if isinstance(obj_id, int) else len(obj_id)
        img = cv2.imread(os.path.join(self.db_root_dir, img_path))
        img = np.array(img, dtype=np.float32)
        if self.rgb:
            img = img[:, :, [2, 1, 0]]

        if label_path is not None:
            label = Image.open(os.path.join(self.db_root_dir, label_path))
        else:
            if self.batch_gt:
                gt = np.zeros(
                    np.append(img.shape[:-1], n_obj), dtype=np.float32)
            else:
                gt = np.zeros(img.shape[:-1], dtype=np.float32)

        if label_path is not None:
            gt_tmp = np.array(label, dtype=np.uint8)
            if self.batch_gt:
                gt = np.zeros(np.append(n_obj, gt_tmp.shape), dtype=np.float32)
                for ii, k in enumerate(obj_id):
                    gt[ii, :, :] = gt_tmp == k
                gt = gt.transpose((1, 2, 0))
            else:
                gt = (gt_tmp == obj_id).astype(np.float32)

        if self.batch_gt:
            prev_round_mask = np.zeros(
                np.append(img.shape[:-1], n_obj), dtype=np.float32)
            for ii, k in enumerate(obj_id):
                prev_round_mask[:, :, ii] = prev_round_mask_tmp == k
        else:
            prev_round_mask = (prev_round_mask_tmp ==
                               obj_id).astype(np.float32)

        return img, gt, prev_round_mask

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))
        return list(img.shape[:2])

    def __str__(self):
        return 'DAVIS2017'


if __name__ == '__main__':
    a = DAVIS2017(split='val',  custom_frames=[21, 22], seq_name='gold-fish', rgb=True,
                  no_gt=False, retname=True, prev_round_masks=np.zeros([40, 480, 854]))
    c = a.__getitem__(0)
    b = 1
