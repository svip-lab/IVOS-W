import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.momory_pool import ReplayMemory


class Brain(nn.Module):
    def __init__(self, lstm_input_channels=128, hidden_channels=128, num_fc_concat=128):
        super(Brain, self).__init__()
        self.input_channels = lstm_input_channels
        self.hidden_channels = hidden_channels
        self.num_fc_concat = num_fc_concat

        self.encoder_fc1 = nn.Linear(2, 128)
        # self.encoder_fc1 = nn.Linear(1, 128)
        self.encoder_fc2 = nn.Linear(128, self.input_channels)

        self.lstm_cell = nn.LSTMCell(
            self.input_channels, self.hidden_channels, False)

        self.decoder_fc1 = nn.Linear(
            2 * self.hidden_channels, self.num_fc_concat)
        self.decoder_fc2 = nn.Linear(self.num_fc_concat, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        '''
        :param input_: N x T x P
        :return:
        '''
        N, T, P = input.shape

        final_state_fw = []
        final_state_bw = []
        final_Q_outputs = []
        state_fw = None
        state_bw = None
        input_ = input.transpose(0, 1)
        for t in range(T):
            # forward input throught all convLSTM layers
            # input_: N x C x P
            fw_feature = self.encoder_fc2(self.relu(self.encoder_fc1(input_[t]))).view(N, -1)
            bw_feature = self.encoder_fc2(self.relu(self.encoder_fc1(input_[T - 1 - t]))).view(N, -1)
            state_fw = self.lstm_cell(fw_feature, state_fw)
            state_bw = self.lstm_cell(bw_feature, state_bw)
            final_state_fw.append(state_fw[0].clone())
            final_state_bw.append(state_bw[0].clone())

        # calculate final output
        final_state_bw.reverse()
        for t in range(T):
            # calculate Q_value of t frame by final hidden state
            final_state = torch.cat([final_state_fw[t], final_state_bw[t]], 1)
            final_Q_outputs.append(self.decoder_fc2(
                self.relu(self.decoder_fc1(self.relu(final_state)))))

        return torch.cat(final_Q_outputs, 1)


class Agent(nn.Module):
    def __init__(self, device, cfg):
        super(Agent, self).__init__()
        self.cfg = cfg
        self.device = device
        self.memory_size = self.cfg.agent.memory_size
        self.GAMMA = self.cfg.agent.gamma
        self.EPS_START = self.cfg.agent.eps_start
        self.EPS_END = self.cfg.agent.eps_end
        self.EPS_DECAY = self.cfg.agent.eps_decay
        self.steps_done = 0
        self.update_rate = self.cfg.agent.update_rate

        self.subset = self.cfg.data.subset
        self.memory_pool = ReplayMemory(self.memory_size)

        self.policy_net = Brain()
        self.target_net = Brain()

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        # if torch.cuda.is_available():
        #     self.policy_net = torch.nn.DataParallel(self.policy_net)
        #     self.target_net = torch.nn.DataParallel(self.target_net)

        self.loss = []
        self.loss_position = 0
        self.loss_capacity = 32
        self.loss_avg = 0

        params = self.policy_net.parameters()

        self.optimizer = optim.Adam(params, lr=cfg.agent.lr, weight_decay=cfg.agent.weight_decay)

    def update_agent(self, sample):
        if sample is None:
            print('no input')
            return
        action = sample['action'].view(
            sample['action'].shape[0], -1).to(self.device)
        batch_size = sample['action'].shape[0]
        reward_step = sample['reward_step'].float().view(
            batch_size, -1).to(self.device)
        reward_done = sample['reward_done'].float().view(
            batch_size, -1).to(self.device)
        done = sample['done'].float().view(
            batch_size, -1).float().to(self.device)
        old_state_iou = sample['old_state_iou'].float().view(
            batch_size, -1).float().to(self.device)
        new_state_iou = sample['new_state_iou'].float().view(
            batch_size, -1).float().to(self.device)
        annotated_frames = sample['annotated_frames'].float().view(
            batch_size, -1).float().to(self.device)
        next_annotated_frames = sample['next_annotated_frames'].float().view(
            batch_size, -1).float().to(self.device)
        GAMMA = torch.from_numpy(
            np.array(self.GAMMA, dtype=np.float32)).to(self.device)


        state = torch.stack([old_state_iou, annotated_frames], 2)
        new_state = torch.stack([new_state_iou, next_annotated_frames], 2)

        scale_factor_step = 0.1
        scale_factor_done = 0.1
        with torch.no_grad():
            self.set_eval()
            output = self.policy_net(new_state)
            next_action = output.max(1)[1].view(batch_size, -1)
            Q_next_state_target = self.target_net(new_state)
            Q_next_state = Q_next_state_target.gather(1, next_action.long()).detach()
            Q_state_action_target_step = (Q_next_state * GAMMA) + reward_step * scale_factor_step
            Q_state_action_target_done = reward_done * scale_factor_done
            self.set_train()

        # formulate bellman equation
        Q_state = self.policy_net(state)
        Q_state_action = Q_state.gather(1, action.long())

        # ====== update policy net ======

        loss_step = F.mse_loss(Q_state_action, Q_state_action_target_step)
        loss_done = F.mse_loss(Q_state_action, Q_state_action_target_done)
        loss = loss_step + loss_done


        self.optimizer.zero_grad()
        loss.backward()
        self._update_avg_loss(loss)
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # ====== update target net ======
        if np.random.random() < self.update_rate:
            print("target_net updated!")
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()

    def action(self, state, verbose=True):
        self.steps_done += 1
        if not self.cfg.phase == 'train':
            eps_threshold = 0
        else:  # epsilon-greedy
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                math.exp(-0.5 * self.steps_done / self.EPS_DECAY)

        inputs = torch.Tensor(state[np.newaxis]).to(self.device)

        rand_flag = random.random()
        if rand_flag > eps_threshold:
            if verbose:
                print(f"step:{self.steps_done}, rand_flag:{rand_flag:.4f}, eps_threshold:{eps_threshold:.4f}, "
                      f"frame index was selected by agent")
            with torch.no_grad():
                self.policy_net.eval()
                state_output = self.policy_net(inputs)
                self.policy_net.train()
            Q_value_numpy = state_output.to("cpu").detach().numpy().squeeze()
            action = Q_value_numpy.argmax()
            return action
        else:
            if verbose:
                print(f"step:{self.steps_done}, rand_flag:{rand_flag:.4f}, eps_threshold:{eps_threshold:.4f}, "
                      f"frame index was selected randomly")
            action_idx = np.array(range(inputs.shape[1]))
            action = random.choice(action_idx)
            return action

    def _update_avg_loss(self, loss):
        if len(self.loss) < self.loss_capacity:
            self.loss.append(None)
        self.loss[self.loss_position] = float(loss.detach().to("cpu").numpy())
        self.loss_position = (self.loss_position + 1) % self.loss_capacity
        self.loss_avg = sum(self.loss) / len(self.loss)

    def set_train(self):
        self.policy_net.train()
        self.target_net.train()

    def set_eval(self):
        self.policy_net.eval()
        self.target_net.eval()

    def memory(self, state,
               old_frame,
               next_state,
               reward_step,
               reward_done,
               is_done,
               state_iou,
               next_state_iou,
               annotated_frames_str,
               next_annotated_frames_str,
               report_save_dir):
        self.memory_pool.push(state,
                              old_frame,
                              next_state,
                              reward_step,
                              reward_done,
                              is_done,
                              state_iou,
                              next_state_iou,
                              annotated_frames_str,
                              next_annotated_frames_str)
        self.memory_pool.push_to_csv(report_save_dir)

    def get_avg_loss(self):
        return self.loss_avg