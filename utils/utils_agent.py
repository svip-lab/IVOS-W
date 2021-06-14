import copy
import numpy as np

import torch


def goal_only_reward(sequence, n_interaction, scribble_iter, repeat_selection, iou_new, df=None):

    reward_step = np.array(1) if not repeat_selection else np.array(-1)

    if df is not None:
        df = df[df.sequence == sequence]
        df = df[df.n_interaction_next == n_interaction]
        prev_metric = df[((df.scribble_iter - 1) % 3)
                         == ((scribble_iter - 1) % 3)]

        prev_metric = [np.array([float(i) for i in item.split('/')]).mean()
                       for item in prev_metric.next_state_iou]
        prev_metric = np.array(prev_metric)
        assert len(prev_metric) == 30

        metric = iou_new.mean()

        mean, std = prev_metric.mean(), prev_metric.std(ddof=1)

        # # Eq.2
        # reward_done = (metric - mean) / std

        # # Eq.3
        reward_done = (metric - mean - std) / std

    else:
        reward_done = np.array(0)

    return reward_step, reward_done


def select_next_frame(frame_value, metric='min', prev_frames=None):

    nb_frames = len(frame_value)

    if metric == 'random':
        return int(np.random.randint(nb_frames, size=1))

    if metric == 'uniform':
        assert prev_frames is not None

    if metric == 'prob':
        temp_prob = np.random.rand()
        prob = F.softmax(torch.Tensor(frame_value), 0)
        k = 0
        while (temp_prob > 0):
            temp_prob = temp_prob - prob[k]
            k += 1
        frame_to_annotate = (k - 1)

        return frame_to_annotate

    if metric == 'max':
        frame_value = -frame_value

    if prev_frames is not None:
        value_idx = frame_value.argsort()
        i = 0
        while i < nb_frames and value_idx[i] in prev_frames:
            i += 1
        if i == nb_frames:
            return frame_value.argmin()  # All the frames have been annotated
        frame_to_annotate = value_idx[i]

    else:
        frame_to_annotate = frame_value.argmin()

    return frame_to_annotate


def recommend_frame(cfg_yl, assess_net, agent, device, n_frame, n_objects, all_F, all_P, new_masks_quality, prev_frames,
                    annotated_frames_list, mask_quality, first_frame, max_nb_interactions):
    if cfg_yl.setting == 'oracle':
        if cfg_yl.method == 'worst':
            next_frame = select_next_frame(new_masks_quality, metric='worst', prev_frames=prev_frames)
        elif cfg_yl.method == 'ours':
            mask_quality = new_masks_quality
            annotated_frames_list_np = np.zeros(len(new_masks_quality))
            for i in annotated_frames_list: annotated_frames_list_np[i] += 1
            state = np.stack([mask_quality, annotated_frames_list_np], 1)
            with torch.no_grad():
                next_frame = agent.action(state)
        else:
            raise NotImplementedError
    elif cfg_yl.setting == 'wild':
        if cfg_yl.method == 'random':
            next_frame = select_next_frame(new_masks_quality, metric='random')
        elif cfg_yl.method == 'linspace':
            next_frame = prev_frames[0]
            len_subseq = min(max_nb_interactions, n_frame)
            subseq = gen_subseq(first_frame, n_frame, len_subseq, 'equal')
            for i in subseq:
                if i not in prev_frames:
                    next_frame = i
                    break
        elif cfg_yl.method == 'worst':
            mask_quality_pred = np.zeros((n_frame, n_objects))
            all_F = all_F.to(device)
            all_P = all_P.to(device)
            with torch.no_grad():
                for i in range(n_objects):
                    mask_quality_pred[:, i:i + 1] = assess_net(all_F, all_P[:, i + 1]).cpu().numpy()
            mask_quality[:] = mask_quality_pred.mean(1)
            next_frame = select_next_frame(mask_quality, metric='worst', prev_frames=prev_frames)
        elif cfg_yl.method == 'ours':
            annotated_frames_list_np = np.zeros(len(new_masks_quality))
            for i in annotated_frames_list: annotated_frames_list_np[i] += 1
            all_F = all_F.to(device)
            all_P = all_P.to(device)
            mask_quality_pred = np.zeros((n_frame, n_objects))
            with torch.no_grad():
                for i in range(n_objects):
                    mask_quality_pred[:, i:i + 1] = assess_net(all_F, all_P[:, i + 1]).cpu().numpy()
                mask_quality[:] = mask_quality_pred.mean(1)
                state = np.stack([mask_quality, annotated_frames_list_np], 1)
                next_frame = agent.action(state)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return next_frame


def gen_subseq(first_frame, n_frame, len_subseq, subseq_style='consecutive'):
    if subseq_style == 'consecutive':
        assert n_frame >= len_subseq
        i_start = max(0, (first_frame - len_subseq + 1))
        i_end = first_frame - max((first_frame + len_subseq) - n_frame, 0)
        i = int((i_start + i_end) / 2)
        subseq = list(range(i, i + len_subseq))
    elif subseq_style == 'equal':
        start = 0
        end = n_frame - 1
        if (end - start + 1) < len_subseq + 1:
            subseq = list(np.array(range(len_subseq)))
        else:
            assert (end - start + 1) >= len_subseq + 1
            subseq = np.linspace(start, n_frame-1, num=len_subseq + 1).astype(int)
            while True:
                if first_frame not in list(subseq):
                    subseq += 1
                else:
                    break
            if first_frame != subseq[-1]:
                subseq = list(subseq[:-1])
            else:
                subseq = list(subseq[1:])
    else:
        raise NotImplementedError
    return subseq


def agent_train_data_collection(agent,
                                reward_step,
                                reward_done,
                                annotated_frames_list_np,
                                next_annotated_frames_list_np,
                                old_masks_IoU,
                                new_masks_IoU,
                                old_masks_meta,
                                new_masks_meta,
                                done,
                                old_frame,
                                report_save_dir):

    # --- masks_info as state ---
    state = old_masks_meta
    next_state = new_masks_meta
    state_iou = ''
    next_state_iou = ''
    annotated_frames_str = ''
    next_annotated_frames_str = ''
    for i in range(len(old_masks_IoU)):
        state_iou_i_o = old_masks_IoU[i]
        next_state_iou_i_o = new_masks_IoU[i]
        annotated_frame_i_o = annotated_frames_list_np[i]
        next_annotated_frame_i_o = next_annotated_frames_list_np[i]
        state_iou += str(state_iou_i_o) + \
            '/' if i < len(old_masks_IoU) - 1 else str(state_iou_i_o)
        next_state_iou += str(next_state_iou_i_o) + \
            '/' if i < len(old_masks_IoU) - 1 else str(next_state_iou_i_o)
        annotated_frames_str += str(annotated_frame_i_o) + '/' if i < len(
            old_masks_IoU) - 1 else str(annotated_frame_i_o)
        next_annotated_frames_str += str(next_annotated_frame_i_o) + '/' if i < len(
            old_masks_IoU) - 1 else str(next_annotated_frame_i_o)

    agent.memory(state,
                 old_frame,
                 next_state,
                 reward_step,
                 reward_done,
                 done,
                 state_iou,
                 next_state_iou,
                 annotated_frames_str,
                 next_annotated_frames_str,
                 report_save_dir)


def agent_business(cfg_yl, agent, max_nb_interactions, n_interaction, first_scribble, old_masks_metric, new_masks_metric, old_frame, sequence,
                   seen_seq, repeat_selection, df, annotated_frames_list, next_frame, old_masks_meta, new_masks_meta, report_save_dir,
                   agent_train_loader):
    agent_loss_iter = np.array(0)
    reward_step = np.array(0)
    reward_done = np.array(0)
    if not first_scribble and not cfg_yl.phase == 'eval':
        # --- set state, next_state, action and memory all variants ---
        reward_step, reward_done = goal_only_reward(sequence,
                                                     n_interaction,
                                                     seen_seq[sequence],
                                                     repeat_selection,
                                                     new_masks_metric, df=df)

        next_annotated_frames_list_np = np.zeros(len(new_masks_metric))
        annotated_frames_list_np = np.zeros(len(new_masks_metric))
        next_annotated_frames_list = copy.deepcopy(annotated_frames_list)
        next_annotated_frames_list.append(next_frame)
        done = False if (n_interaction < max_nb_interactions) else True
        for i in annotated_frames_list:
            annotated_frames_list_np[i] += 1
        for i in next_annotated_frames_list:
            next_annotated_frames_list_np[i] += 1
        agent_train_data_collection(agent,
                                    reward_step,
                                    reward_done,
                                    annotated_frames_list_np,
                                    next_annotated_frames_list_np,
                                    old_masks_metric,
                                    new_masks_metric,
                                    old_masks_meta,
                                    new_masks_meta,
                                    done,
                                    old_frame,
                                    report_save_dir)

        # train agent
        if n_interaction == max_nb_interactions:
            if cfg_yl.phase == 'train':
                agent_loss_iter_list = []
                for i, sample in enumerate(agent_train_loader):
                    if i == (max_nb_interactions * 3) - 1:
                        break
                    loss = agent.update_agent(sample)
                    agent_loss_iter_list.append(loss)
                agent_loss_iter = np.array(agent_loss_iter_list).mean()
        # reward_step = reward_step.mean()
        # reward_done = reward_done.mean()

    return agent_loss_iter, reward_step, reward_done
