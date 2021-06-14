import json

import torch
import torch.nn as nn

from davisinteractive.dataset.davis import Davis

from config import cfg


def load_network(net,pretrained_dict):

        #pretrained_dict = pretrained_dict
    model_dict = net.state_dict()
           # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)


def rough_ROI(ref_scribble_labels):
    #### b*1*h*w
    dist = 20
    b, _, h, w = ref_scribble_labels.size()
    filter_ = torch.zeros_like(ref_scribble_labels)
    to_fill = torch.zeros_like(ref_scribble_labels)
    for i in range(b):
        no_background = (ref_scribble_labels[i] != -1)
        no_background = no_background.squeeze(0)

        no_b = no_background.nonzero()
        (h_min, w_min), _ = torch.min(no_b, 0)
        (h_max, w_max), _ = torch.max(no_b, 0)

        filter_[i, 0, max(h_min - dist, 0):min(h_max + dist, h - 1), max(w_min - dist, 0):min(w_max + dist, w - 1)] = 1

    final_scribble_labels = torch.where(filter_.byte(), ref_scribble_labels, to_fill)
    return final_scribble_labels

def preprocess(db_root_dir, seqs, seq_list_file):
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

def get_results(model, ref_frame_embedding, scribble_label, prev_label, eval_global_map_tmp_dic, local_map_dics,
                n_interaction, sequence, obj_nums, next_frame, first_scribble, h, w, prev_label_storage, total_frame_num, embedding_memory):
    pred_masks = []
    pred_masks_reverse = []
    probs = []
    probs_reverse = []
    tmp_dic, local_map_dics = model.int_seghead(ref_frame_embedding=ref_frame_embedding,
                                                ref_scribble_label=scribble_label,
                                                prev_round_label=prev_label,
                                                global_map_tmp_dic=eval_global_map_tmp_dic,
                                                local_map_dics=local_map_dics,
                                                interaction_num=n_interaction,
                                                seq_names=[sequence],
                                                gt_ids=torch.Tensor([obj_nums]),
                                                frame_num=[next_frame],
                                                first_inter=first_scribble)
    pred_label = tmp_dic[sequence]
    pred_label = nn.functional.interpolate(pred_label, size=(h, w), mode='bilinear',
                                           align_corners=True)
    probs.append(pred_label)
    pred_label = torch.argmax(pred_label, dim=1)
    pred_masks.append(pred_label.float())
    prev_label_storage[next_frame] = pred_label

    ref_prev_label = pred_label.unsqueeze(0)
    prev_label = pred_label.unsqueeze(0)
    prev_embedding = ref_frame_embedding
    #### Propagation ->
    for ii in range(next_frame + 1, total_frame_num):
        current_embedding = embedding_memory[ii]
        current_embedding = current_embedding.unsqueeze(0)

        prev_label = prev_label.cuda()
        tmp_dic, eval_global_map_tmp_dic, local_map_dics = model.prop_seghead(ref_frame_embedding,
                                                                              prev_embedding,
                                                                              current_embedding,
                                                                              scribble_label,
                                                                              prev_label,
                                                                              normalize_nearest_neighbor_distances=True,
                                                                              use_local_map=True,
                                                                              seq_names=[sequence],
                                                                              gt_ids=torch.Tensor([obj_nums]),
                                                                              k_nearest_neighbors=cfg.KNNS,
                                                                              global_map_tmp_dic=eval_global_map_tmp_dic,
                                                                              local_map_dics=local_map_dics,
                                                                              interaction_num=n_interaction,
                                                                              start_annotated_frame=next_frame,
                                                                              frame_num=[ii],
                                                                              dynamic_seghead=model.dynamic_seghead)
        pred_label = tmp_dic[sequence]
        pred_label = nn.functional.interpolate(pred_label, size=(h, w), mode='bilinear',
                                               align_corners=True)
        probs.append(pred_label)

        pred_label = torch.argmax(pred_label, dim=1)
        pred_masks.append(pred_label.float())
        prev_label = pred_label.unsqueeze(0)
        prev_embedding = current_embedding
        prev_label_storage[ii] = pred_label

    prev_label = ref_prev_label
    prev_embedding = ref_frame_embedding
    #######
    # Propagation <-
    for ii in range(next_frame):
        current_frame_num = next_frame - 1 - ii
        current_embedding = embedding_memory[current_frame_num]
        current_embedding = current_embedding.unsqueeze(0)
        prev_label = prev_label.cuda()
        tmp_dic, eval_global_map_tmp_dic, local_map_dics = model.prop_seghead(ref_frame_embedding,
                                                                              prev_embedding,
                                                                              current_embedding,
                                                                              scribble_label,
                                                                              prev_label,
                                                                              normalize_nearest_neighbor_distances=True,
                                                                              use_local_map=True,
                                                                              seq_names=[sequence],
                                                                              gt_ids=torch.Tensor([obj_nums]),
                                                                              k_nearest_neighbors=cfg.KNNS,
                                                                              global_map_tmp_dic=eval_global_map_tmp_dic,
                                                                              local_map_dics=local_map_dics,
                                                                              interaction_num=n_interaction,
                                                                              start_annotated_frame=next_frame,
                                                                              frame_num=[
                                                                                  current_frame_num],
                                                                              dynamic_seghead=model.dynamic_seghead)
        pred_label = tmp_dic[sequence]
        pred_label = nn.functional.interpolate(pred_label, size=(h, w), mode='bilinear',
                                               align_corners=True)
        probs_reverse.append(pred_label)
        pred_label = torch.argmax(pred_label, dim=1)
        pred_masks_reverse.append(pred_label.float())
        prev_label = pred_label.unsqueeze(0)
        prev_embedding = current_embedding
        ####
        prev_label_storage[current_frame_num] = pred_label
    pred_masks_reverse.reverse()
    pred_masks_reverse.extend(pred_masks)
    probs_reverse.reverse()
    probs_reverse.extend(probs)

    final_masks = torch.cat(pred_masks_reverse, 0)
    all_P = torch.softmax(torch.cat(probs_reverse, 0), 1)

    return final_masks, all_P
