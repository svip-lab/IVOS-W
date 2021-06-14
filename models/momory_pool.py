import os
import random
from collections import namedtuple

import numpy as np
import pandas as pd

Transition = namedtuple('Transition', ('state',
                                       'action',
                                       'next_state',
                                       'reward_step',
                                       'reward_done',
                                       'done',
                                       'state_iou',
                                       'next_state_iou',
                                       'annotated_frames',
                                       'next_annotated_frames'
                                       ))


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = -1
        self.basename_csv = 'memory_pool.csv'

        self.COLUMNS = [
            'sequence',
            'scribble_iter',
            'n_interaction',
            'n_interaction_next',
            'action',
            'reward_step',
            'reward_done',
            'done',
            'state_iou',
            'next_state_iou',
            'annotated_frames',
            'next_annotated_frames'
        ]
        self.memory_pd = pd.DataFrame(columns=self.COLUMNS)

    def load_from_csv(self, path_to_random_memory_csv, report_save_dir=None, sample_th=0):

        df = pd.read_csv(path_to_random_memory_csv, index_col=0)
        df = df[:self.capacity]

        # filter the rubbish sequence
        seq_names_all = np.array(df['sequence'].values.tolist())
        _, seq_names_idx = np.unique(seq_names_all, return_index=True)
        seq_names = [seq_names_all[i]
                     for i in range(len(seq_names_all)) if i in seq_names_idx]
        if sample_th > 0:
            assert sample_th < 1
            self.seq_list = []
            for i, seq in enumerate(seq_names):
                mp_seq = df[df.sequence == seq]
                if len(mp_seq) == 0:
                    continue
                state_iou = mp_seq.state_iou.values.tolist()
                next_state_iou = mp_seq.next_state_iou.values.tolist()
                p_min = np.array([p.split('/') for p in state_iou]
                                 ).astype(np.float).mean(1).min()
                p_max = np.array(
                    [p.split('/') for p in next_state_iou]).astype(np.float).mean(1).max()
                if p_max - p_min > sample_th:
                    self.seq_list.append(seq)
            print(
                f"the number of available samples under threshold {sample_th}: {len(self.seq_list)}")
        else:
            for i, seq in enumerate(seq_names):
                self.seq_list.append(seq)

        sequence = df['sequence'].tolist()
        scribble_iter = df['scribble_iter'].tolist()
        n_interaction = df['n_interaction'].tolist()
        n_interaction_next = df['n_interaction_next'].tolist()
        action = df['action'].tolist()
        reward_step = df['reward_step'].tolist()
        reward_done = df['reward_done'].tolist()
        done = df['done'].tolist()
        state_iou = df['state_iou'].tolist()
        next_state_iou = df['next_state_iou'].tolist()
        annotated_frames = df['annotated_frames'].tolist()
        next_annotated_frames = df['next_annotated_frames'].tolist()

        capacity = 0
        for i in range(min(len(sequence), self.capacity)):
            if sample_th > 0:
                assert len(self.seq_list) > 0
                if not sequence[i] in self.seq_list:
                    continue
            capacity += 1
            state = dict(
                sequence=sequence[i], scribble_iter=scribble_iter[i], n_interaction=n_interaction[i])
            next_state = dict(
                sequence=sequence[i], scribble_iter=scribble_iter[i], n_interaction=n_interaction_next[i])

            self.push(state,
                      action[i],
                      next_state,
                      reward_step[i],
                      reward_done[i],
                      done[i],
                      state_iou[i],
                      next_state_iou[i],
                      annotated_frames[i],
                      next_annotated_frames[i])
        self.capacity = capacity

        if not os.path.exists(report_save_dir):
            os.makedirs(report_save_dir)
        csv_path = os.path.join(report_save_dir, self.basename_csv)
        self.memory_pd = df[:self.capacity]
        self.memory_pd.to_csv(csv_path)

    def push(self, *args):
        # saves a transition
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.position = (self.position + 1) % self.capacity
        self.memory[self.position] = Transition(*args)
        # make a position loop

    def push_to_csv(self, report_save_dir):
        # saves a transition
        memory = Transition(*zip(*self.memory))
        csv_path = os.path.join(report_save_dir, self.basename_csv)

        sample = {}
        sample['sequence'] = [memory.state[self.position]['sequence']]
        sample['scribble_iter'] = [
            memory.state[self.position]['scribble_iter']]
        sample['n_interaction'] = [
            memory.state[self.position]['n_interaction']]
        sample['n_interaction_next'] = [
            memory.next_state[self.position]['n_interaction']]
        sample['action'] = [memory.action[self.position]]
        sample['reward_step'] = [memory.reward_step[self.position]]
        sample['reward_done'] = [memory.reward_done[self.position]]
        sample['done'] = [memory.done[self.position]]
        sample['state_iou'] = [memory.state_iou[self.position]]
        sample['next_state_iou'] = [memory.next_state_iou[self.position]]
        sample['annotated_frames'] = [memory.annotated_frames[self.position]]
        sample['next_annotated_frames'] = [
            memory.next_annotated_frames[self.position]]

        sample = pd.DataFrame(data=sample, columns=self.COLUMNS)
        self.memory_pd = pd.concat([self.memory_pd, sample], ignore_index=True)
        self.memory_pd = self.memory_pd.drop(self.memory_pd.index.min()) if len(
            self.memory_pd) > self.capacity else self.memory_pd
        self.memory_pd.to_csv(csv_path)

    def random_sample(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        transitions = random.sample(self.memory, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)
