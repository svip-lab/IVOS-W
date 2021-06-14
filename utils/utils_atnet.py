import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.davis_dataset import DAVIS2017

from libs import custom_transforms as tr
from libs import utils, utils_torch


def run_VOS_singleiact(net, config, split, scribbles_data, annotated_frames, final_masks, num_frames, n_objects, n_interaction, subseq,
                       pad_info, anno_3chEnc_r5_list, anno_6chEnc_r5_list, prob_map_of_frames, hpad1, hpad2, wpad1, wpad2):

    annotated_frames_np = np.array(annotated_frames)
    num_workers = 4
    annotated_now = annotated_frames[-1]
    scribbles_list = scribbles_data['scribbles']
    seq_name = scribbles_data['sequence']

    output_masks = final_masks.copy().astype(np.float64)

    prop_list = utils.get_prop_list(
        annotated_frames, annotated_now, num_frames, proportion=config.test_propagation_proportion)
    prop_fore = sorted(prop_list)[0]
    prop_rear = sorted(prop_list)[-1]

    # Interaction settings
    pm_ps_ns_3ch_t = []  # n_obj,3,h,w
    if n_interaction == 1:
        for obj_id in range(1, n_objects + 1):
            pos_scrimg = utils.scribble_to_image(scribbles_list, annotated_now, obj_id,
                                                 dilation=config.scribble_dilation_param,
                                                 prev_mask=final_masks[annotated_now])
            pm_ps_ns_3ch_t.append(np.stack(
                [np.ones_like(pos_scrimg) / 2, pos_scrimg, np.zeros_like(pos_scrimg)], axis=0))
        pm_ps_ns_3ch_t = np.stack(pm_ps_ns_3ch_t, axis=0)  # n_obj,3,h,w

    else:
        for obj_id in range(1, n_objects + 1):
            prev_round_input = (final_masks[annotated_now] == obj_id).astype(
                np.float32)  # H,W
            pos_scrimg, neg_scrimg = utils.scribble_to_image(scribbles_list, annotated_now, obj_id,
                                                             dilation=config.scribble_dilation_param,
                                                             prev_mask=final_masks[annotated_now], blur=True,
                                                             singleimg=False, seperate_pos_neg=True)
            pm_ps_ns_3ch_t.append(
                np.stack([prev_round_input, pos_scrimg, neg_scrimg], axis=0))
        pm_ps_ns_3ch_t = np.stack(pm_ps_ns_3ch_t, axis=0)  # n_obj,3,h,w
    pm_ps_ns_3ch_t = torch.from_numpy(pm_ps_ns_3ch_t).cuda()

    if (prop_list[0] != annotated_now) and (prop_list.count(annotated_now) != 2):
        # print(str(prop_list))
        raise NotImplementedError
    # print(str(prop_list))  # we made our proplist first backward, and then forward

    composed_transforms = transforms.Compose(
        [tr.Normalize_ApplymeanvarImage(config.mean, config.var),
        tr.ToTensor()]
    )
    db_test = DAVIS2017(
        split=split, subseq=subseq, transform=composed_transforms, 
        root=config.davis_dataset_dir, custom_frames=prop_list, seq_name=seq_name, 
        rgb=True, obj_id=None, no_gt=True, retname=True, prev_round_masks=final_masks)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    flag = 0  # 1: propagating backward, 2: propagating forward
    # print('[{:01d} round] processing...'.format(n_interaction))

    for ii, batched in enumerate(testloader):
        # batched : image, scr_img, 0~fr, meta
        inpdict = dict()
        operating_frame = int(batched['meta']['frame_id'][0])

        for inp in batched:
            if inp == 'meta':
                continue
            inpdict[inp] = Variable(batched[inp]).cuda()
        inpdict['image'] = inpdict['image'].expand(n_objects, -1, -1, -1)

        #################### Iaction ########################
        if operating_frame == annotated_now:  # Check the round is on interaction
            if flag == 0:
                flag += 1
                adjacent_to_anno = True
            elif flag == 1:
                flag += 1
                adjacent_to_anno = True
                continue
            else:
                raise NotImplementedError

            pm_ps_ns_3ch_t = torch.nn.ReflectionPad2d(
                pad_info[1] + pad_info[0])(pm_ps_ns_3ch_t)
            inputs = torch.cat([inpdict['image'], pm_ps_ns_3ch_t], dim=1)
            # [nobj, 1, P_H, P_W], # [n_obj,2048,h/16,w/16]
            output_logit, anno_6chEnc_r5 = net.forward_ANet(inputs)
            output_prob_anno = torch.sigmoid(output_logit)
            prob_onehot_t = output_prob_anno[:, 0].detach()

            anno_3chEnc_r5, _, _, r2_prev_fromanno = net.encoder_3ch.forward(
                inpdict['image'])
            anno_6chEnc_r5_list.append(anno_6chEnc_r5)
            anno_3chEnc_r5_list.append(anno_3chEnc_r5)

            if len(anno_6chEnc_r5_list) != len(annotated_frames):
                raise NotImplementedError

        #################### Propagation ########################
        else:
            # Flag [1: propagating backward, 2: propagating forward]
            if adjacent_to_anno:
                r2_prev = r2_prev_fromanno
                predmask_prev = output_prob_anno
            else:
                predmask_prev = output_prob_prop
            adjacent_to_anno = False

            output_logit, r2_prev = net.forward_TNet(anno_3chEnc_r5_list, inpdict['image'], anno_6chEnc_r5_list,
                                                     r2_prev, predmask_prev)  # [nobj, 1, P_H, P_W]
            output_prob_prop = torch.sigmoid(output_logit)
            prob_onehot_t = output_prob_prop[:, 0].detach()

            smallest_alpha = 0.5
            if flag == 1:
                sorted_frames = annotated_frames_np[annotated_frames_np <
                                                    annotated_now]
                if len(sorted_frames) == 0:
                    alpha = 1
                else:
                    closest_addianno_frame = np.max(sorted_frames)
                    alpha = smallest_alpha + (1 - smallest_alpha) * (
                        (operating_frame - closest_addianno_frame) / (annotated_now - closest_addianno_frame))
            else:
                sorted_frames = annotated_frames_np[annotated_frames_np >
                                                    annotated_now]
                if len(sorted_frames) == 0:
                    alpha = 1
                else:
                    closest_addianno_frame = np.min(sorted_frames)
                    alpha = smallest_alpha + (1 - smallest_alpha) * (
                        (closest_addianno_frame - operating_frame) / (closest_addianno_frame - annotated_now))

            prob_onehot_t = (alpha * prob_onehot_t) + \
                ((1 - alpha) * prob_map_of_frames[operating_frame])

        # Final mask indexing
        prob_map_of_frames[operating_frame] = prob_onehot_t

    output_masks[prop_fore:prop_rear + 1] = \
        utils_torch.combine_masks_with_batch(prob_map_of_frames[prop_fore:prop_rear + 1],
                                             n_obj=n_objects, th=config.test_propth
                                             )[:, 0, hpad1:-hpad2, wpad1:-wpad2].cpu().numpy().astype(np.float)  # f,h,w

    torch.cuda.empty_cache()
    bg_dummy = torch.zeros_like(prob_map_of_frames[:, 0:1])
    all_P = torch.cat([bg_dummy, prob_map_of_frames], 1)[:, :, hpad1:-hpad2, wpad1:-wpad2]

    return output_masks, all_P