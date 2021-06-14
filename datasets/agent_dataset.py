import json
import os
import time

import numpy as np
import pandas as pd
import torch
from davisinteractive.dataset import Davis


def preprocess(db_root_dir, seqs, seq_list_file):
    seq_dict = {}
    for seq in seqs:
        # Read object masks and get number of objects
        n_obj = Davis.dataset[seq]['num_objects']

        seq_dict[seq] = list(range(1, n_obj + 1))

    print('Preprocessing finished')

    return seq_dict


class DAVIS2017AgentTrain(torch.utils.data.Dataset):
    def __init__(self,
                 split=None,
                 db_root_dir=None,
                 save_result_dir=None,
                 memory_size=None,
                 transform=None,
                 seq_list=None):

        self.seq_list = seq_list
        self.split = split
        self.db_root_dir = db_root_dir
        self.save_result_dir = save_result_dir
        self.memory_size = memory_size
        self.transform = transform

        memory_pool_csv_path = os.path.join(save_result_dir, 'memory_pool.csv')
        assert os.path.exists(
            memory_pool_csv_path), f'{memory_pool_csv_path} does not exist'
        while True:
            try:
                memory_pool = pd.read_csv(
                    memory_pool_csv_path, index_col=0, low_memory=False)
                break
            except:
                print(
                    f"catch some EXCEPTION when try to load {memory_pool_csv_path}")
                time.sleep(10)

        n_sample = memory_pool.shape[0]
        n_sample = min(n_sample, self.memory_size)
        memory_samples = memory_pool.sample(n_sample)
        memory_samples = memory_samples.values.tolist()

        self.seqs = []
        with open(os.path.join(self.db_root_dir, 'ImageSets', '2017', self.split + '.txt')) as f:
            seqs_tmp = f.readlines()
        seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
        self.seqs.extend(seqs_tmp)
        self.seq_list_file = os.path.join(
            self.db_root_dir, 'ImageSets', '2017', self.split + '_instances.txt')

        # Precompute the dictionary with the objects per sequence
        if not self._check_preprocess():
            self.seq_dict = preprocess(
                self.db_root_dir, self.seqs, self.seq_list_file)

        samples_list = []
        for i in range(len(memory_samples)):
            memory = memory_samples[i]
            sequence = memory[0]
            if self.seq_list is not None:
                assert len(self.seq_list) > 0
                if sequence not in self.seq_list:
                    continue
            scribble_iter = memory[1]
            n_interaction = memory[2]
            n_interaction_next = memory[3]
            action = memory[4]
            reward_step = memory[5]
            reward_done = memory[6]
            done = memory[7]
            old_state_iou_str = str(memory[8]).split('/')
            new_state_iou_str = str(memory[9]).split('/')
            annotated_frames_str = str(memory[10]).split('/')
            next_annotated_frames_str = str(memory[11]).split('/')
            old_state_iou = [np.array(iou).astype(
                float)[np.newaxis, np.newaxis] for iou in old_state_iou_str]
            new_state_iou = [np.array(iou).astype(
                float)[np.newaxis, np.newaxis] for iou in new_state_iou_str]
            annotated_frames = [np.array(annotated_frame).astype(
                float)[np.newaxis, np.newaxis] for annotated_frame in annotated_frames_str]
            next_annotated_frames = [np.array(next_annotated_frame).astype(float)[
                np.newaxis, np.newaxis] for next_annotated_frame in next_annotated_frames_str]

            assert sequence in self.seq_dict.keys(
            ), '{} not in {} set.'.format(sequence, self.split)

            sample = dict()
            sample['action'] = action
            sample['old_state_iou'] = np.concatenate(old_state_iou, 1)
            sample['new_state_iou'] = np.concatenate(new_state_iou, 1)
            sample['annotated_frames'] = np.concatenate(annotated_frames, 1)
            sample['next_annotated_frames'] = np.concatenate(
                next_annotated_frames, 1)
            sample['reward_step'] = reward_step
            sample['reward_done'] = reward_done
            sample['done'] = done

            samples_list.append(sample)

        self.samples_list = samples_list

    def _check_preprocess(self):
        if not os.path.isfile(self.seq_list_file):
            return False
        else:
            self.seq_dict = json.load(open(self.seq_list_file, 'r'))
            return True

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        sample = self.samples_list[idx]

        if self.transform is not None:
            sample = self.transform(sample)
        # sample['reward_done'] = torch.Tensor([float(sample['reward_done'])])
        # sample['reward_step'] = torch.Tensor([int(sample['reward_step'])])
        # sample['done'] = torch.Tensor([bool(sample['done'])])
        return sample


def load_agent_dataset(cfg, seq_list):
    train_transform = None

    dataset_root_dir = None
    if cfg.dataset == 'davis':
        dataset_root_dir = cfg.data.root_dir_davis
    elif cfg.dataset == 'youtube_vos':
        dataset_root_dir = cfg.data.root_dir_scribble_youtube_vos
    elif cfg.dataset == 'combine':
        dataset_root_dir = cfg.data.root_dir_combine

    dataset = DAVIS2017AgentTrain(transform=train_transform,
                                  split=cfg.data.subset,
                                  memory_size=cfg.agent.memory_size,
                                  db_root_dir=dataset_root_dir,
                                  save_result_dir=cfg.agent.save_result_dir,
                                  seq_list=seq_list)

    return dataset
