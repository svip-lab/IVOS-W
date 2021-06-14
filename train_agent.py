import os
import sys
import time
import copy
import logging
import numpy as np
import pandas as pd
sys.path.append(os.path.join('VOS', 'ATNet'))

from sacred import Experiment
from easydict import EasyDict as edict

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from davisinteractive.dataset.davis import Davis
from davisinteractive import utils as interactive_utils
from davisinteractive.session import DavisInteractiveSession

from utils.utils_agent import (agent_business, gen_subseq, recommend_frame)
from utils.utils_atnet import run_VOS_singleiact
from models.agent import Agent
from utils.misc import (set_random_seed, AverageMeter, save_agent_checkpoint, sequence_metric)
from datasets.agent_dataset import load_agent_dataset

from config import Config
from networks.atnet import ATnet
from libs import utils

cudnn.benchmark = False
cudnn.deterministic = True


def create_basic_stream_logger(format):
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(format)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


ex = Experiment('train')
ex.add_config('./configs/config.yaml')
ex.logger = create_basic_stream_logger('%(name)s - %(message)s')


def davis_config(run, _log):

    kwargs = dict()

    cfg_yl = edict(run.config)
    cfg_yl.phase = 'train'

    # ====== configs ======
    device = torch.device(f"cuda:{cfg_yl.gpu_id}" if torch.cuda.is_available() else "cpu")

    subset = cfg_yl.data.subset
    dataset_root_dir = cfg_yl.data.root_dir_davis
    max_nb_interactions = int(cfg_yl.davis_interactive.max_nb_interactions)
    max_time = None  # Maximum time per object

    davis = Davis(davis_root=dataset_root_dir)

    set_random_seed(2019)

    # ------ Agent ------
    agent = Agent(device=device, cfg=cfg_yl)

    # ATNet
    config = Config()
    config.davis_dataset_dir = dataset_root_dir
    net = ATnet()
    net.cuda()
    net.eval()
    net.load_state_dict(torch.load(os.path.join('VOS', 'ATNet', config.test_load_state_dir)))

    # Assess_net
    assess_net = None

    save_result_dir = cfg_yl.agent.save_result_dir
    os.makedirs(save_result_dir, exist_ok=True)

    path_to_reward = os.path.join(cfg_yl.agent.save_result_dir, cfg_yl.agent.reward_csv)
    path_to_pretrain = os.path.join(cfg_yl.agent.save_result_dir, cfg_yl.agent.pretrain_csv)

    assert os.path.exists(path_to_reward) and os.path.exists(path_to_pretrain)
    df = pd.read_csv(path_to_reward, index_col=0)
    agent.memory_pool.load_from_csv(path_to_pretrain, save_result_dir, cfg_yl.agent.sample_th)
    seq_list = agent.memory_pool.seq_list
    print(f"init memory pool from {path_to_pretrain}, now the size of memory pool is: {len(agent.memory_pool)}")
    davis.sets[subset] = seq_list
    cfg_yl.setting = 'oracle'
    cfg_yl.method = 'ours'
    cfg_yl.num_epochs = 5
    report_save_dir = save_result_dir

    kwargs['cfg_yl'] = cfg_yl
    kwargs['config'] = config
    kwargs['net'] = net
    kwargs['agent'] = agent
    kwargs['assess_net'] = assess_net
    kwargs['davis'] = davis
    kwargs['dataset_root_dir'] = dataset_root_dir
    kwargs['report_save_dir'] = report_save_dir
    kwargs['subset'] = subset
    kwargs['max_nb_interactions'] = max_nb_interactions
    kwargs['max_time'] = max_time
    kwargs['device'] = device
    kwargs['df'] = df

    return kwargs


@ex.automain
def main(_run, _log):

    kwargs = davis_config(_run, _log)

    auc_meter = AverageMeter()
    metric_at_threshold_meter = AverageMeter()
    seen_seq = {}

    # ====== main loop ======
    for epoch in range(1, kwargs['cfg_yl'].num_epochs+1):
        # 'J', 'F', 'J_AND_F'
        metric_to_optimize = kwargs['cfg_yl'].davis_interactive.metric

        with DavisInteractiveSession(host='localhost', davis_root=kwargs['dataset_root_dir'], subset=kwargs['subset'],
                                     metric_to_optimize=metric_to_optimize,
                                     max_nb_interactions=kwargs['max_nb_interactions'], max_time=kwargs['max_time'],
                                     report_save_dir=kwargs['report_save_dir']) as sess:
            # per object per serquence
            final_mask_iou_seq_obj_scb = AverageMeter()
            final_time_seq_obj_scb = AverageMeter()
            final_recommend_time_seq_obj_scb = AverageMeter()
            final_agent_time_seq_obj_scb = AverageMeter()
            final_seg_time_seq_obj_scb = AverageMeter()
            final_reward_step_seq_obj_scb = AverageMeter()
            final_reward_done_seq_obj_scb = AverageMeter()
            agent_loss = AverageMeter()
            i_seq = 0

            sess.connector.service.robot.min_nb_nodes = kwargs['config'].test_min_nb_nodes
            while sess.next():

                # 1 ------ interaction initial ------
                interaction_tic = time.time()
                init_tic = time.time()
                sequence, scribbles, first_scribble = sess.get_scribbles(only_last=False)
                annotated_frames = interactive_utils.scribbles.annotated_frames(sess.sample_last_scribble)

                if first_scribble:
                    i_seq = i_seq + 1
                    interaction_time = AverageMeter()
                    frame_recommend_time = AverageMeter()
                    segment_time = AverageMeter()
                    agent_time = AverageMeter()

                    assert len(annotated_frames) > 0
                    first_frame = annotated_frames[0]
                    next_frame = annotated_frames[0]
                    reward_step_acc = 0
                    reward_done_acc = 0

                    seen_seq[sequence] = 1 if sequence not in seen_seq.keys() else seen_seq[sequence] + 1

                    gt_masks_original = kwargs['davis'].load_annotations(sequence)
                    nb_objects = kwargs['davis'].dataset[sequence]['num_objects']

                    if (seen_seq[sequence] - 1) % 3 == 0:
                        agent_dataset = load_agent_dataset(kwargs['cfg_yl'], kwargs['agent'].memory_pool.seq_list)
                    agent_train_loader = DataLoader(
                        agent_dataset,
                        batch_size=kwargs['cfg_yl'].agent.train_batch_size,
                        shuffle=True,
                        num_workers=kwargs['cfg_yl'].data.num_workers,
                        pin_memory=True)

                    # make subsequence information
                    len_subseq = min(
                        kwargs['cfg_yl'].data.len_subseq, kwargs['davis'].dataset[sequence]['num_frames'])
                    subseq = gen_subseq(
                        first_frame, kwargs['davis'].dataset[sequence]['num_frames'], len_subseq)
                    _log.info(f'subseq: {subseq}')
                    _log.info(f"first_frame:{first_frame}, subseq:{subseq}")
                    n_frame = len_subseq
                    next_frame = subseq.index(next_frame)
                    gt_masks = gt_masks_original[subseq]
                    prev_frames = [next_frame]
                    annotated_frames_list = [next_frame]

                    # ATNet stuff
                    anno_dict = {'frames': [], 'annotated_masks': [],
                                 'masks_tobe_modified': []}
                    info = kwargs['davis'].dataset[sequence]
                    img_size = info['image_size'][::-1]
                    n_objects = info['num_objects']
                    final_masks = np.zeros([n_frame, img_size[0], img_size[1]])

                    vos_kwargs = dict()
                    vos_kwargs['pad_info'] = utils.apply_pad(final_masks[0])[1]
                    vos_kwargs['hpad1'], vos_kwargs['wpad1'] = vos_kwargs['pad_info'][0][0], \
                        vos_kwargs['pad_info'][1][0]
                    vos_kwargs['hpad2'], vos_kwargs['wpad2'] = vos_kwargs['pad_info'][0][1], \
                        vos_kwargs['pad_info'][1][1]
                    h_ds, w_ds = int((img_size[0] + sum(vos_kwargs['pad_info'][0])) / 4), \
                        int((img_size[1] + sum(vos_kwargs['pad_info'][1])) / 4)
                    vos_kwargs['anno_6chEnc_r5_list'], vos_kwargs['anno_3chEnc_r5_list'] = [
                    ], []
                    vos_kwargs['prob_map_of_frames'] = torch.zeros(
                        (n_frame, n_objects, 4 * h_ds, 4 * w_ds)).cuda()
                    vos_kwargs['num_frames'] = n_frame
                    vos_kwargs['n_objects'] = n_objects
                    vos_kwargs['subseq'] = subseq
                    vos_kwargs['n_interaction'] = 1

                    rec_kwargs = dict()
                    rec_kwargs['n_frame'] = n_frame
                    rec_kwargs['n_objects'] = n_objects
                    rec_kwargs['all_F'] = None
                    rec_kwargs['mask_quality'] = None

                    # dataset and VOS business
                    old_frame = None
                    old_masks_meta = None
                    old_masks_metric = None
                    repeat_selection = None
                else:
                    annotated_frames_list_np = np.zeros(len(new_masks_metric))
                    for i in annotated_frames_list:
                        annotated_frames_list_np[i] += 1
                    repeat_selection = next_frame not in list(
                        np.where(annotated_frames_list_np == annotated_frames_list_np.min())[0])
                    annotated_frames_list.append(next_frame)
                    old_frame = next_frame
                    old_masks_meta = new_masks_meta
                    old_masks_metric = new_masks_metric
                    vos_kwargs['n_interaction'] += 1

                # Where we save annotated frames
                anno_dict['frames'].append(next_frame)
                # mask before modefied at the annotated frame
                anno_dict['masks_tobe_modified'].append(
                    final_masks[next_frame])

                scribbles['annotated_frame'] = next_frame
                scribbles_subseq = [scribbles['scribbles'][i] for i in subseq]
                scribbles['scribbles'] = scribbles_subseq
                init_time = time.time() - init_tic

                # 2 ------ segmentation ------
                segment_tic = time.time()
                with torch.no_grad():
                    final_masks, all_P = run_VOS_singleiact(kwargs['net'], kwargs['config'], kwargs['subset'], scribbles,
                                                            anno_dict['frames'], final_masks, **vos_kwargs)

                new_masks = final_masks
                new_masks_metric = sequence_metric(metric_to_optimize, gt_masks, new_masks, nb_objects)
                segment_time.update(time.time()-segment_tic)

                # 3 ------ frame recommendation ------
                frame_recommend_tic = time.time()

                rec_kwargs['all_P'] = all_P
                rec_kwargs['new_masks_quality'] = new_masks_metric
                rec_kwargs['prev_frames'] = prev_frames
                rec_kwargs['annotated_frames_list'] = copy.deepcopy(annotated_frames_list)
                rec_kwargs['first_frame'] = first_frame
                rec_kwargs['max_nb_interactions'] = kwargs['max_nb_interactions']
                next_frame = recommend_frame(kwargs['cfg_yl'], kwargs['assess_net'], kwargs['agent'], kwargs['device'],
                                             **rec_kwargs)

                prev_frames.append(next_frame)
                frame_recommend_time.update(time.time() - frame_recommend_tic)

                # 4 ------ Submit prediction ------
                new_masks_submit = copy.deepcopy(gt_masks_original)
                new_masks_submit[subseq] = new_masks
                sess.submit_masks(new_masks_submit, next_scribble_frame_candidates=[subseq[next_frame]])


                # 5 ------ agent business ------
                agent_tic = time.time()
                new_masks_meta = dict(
                    sequence=sequence, scribble_iter=seen_seq[sequence], n_interaction=vos_kwargs['n_interaction'])
                agent_kwargs = dict()
                agent_kwargs['first_scribble'] = first_scribble
                agent_kwargs['old_masks_metric'] = old_masks_metric
                agent_kwargs['new_masks_metric'] = new_masks_metric
                agent_kwargs['old_frame'] = old_frame
                agent_kwargs['next_frame'] = next_frame
                agent_kwargs['sequence'] = sequence
                agent_kwargs['seen_seq'] = seen_seq
                agent_kwargs['repeat_selection'] = repeat_selection
                agent_kwargs['df'] = kwargs['df']
                agent_kwargs['annotated_frames_list'] = annotated_frames_list
                agent_kwargs['old_masks_meta'] = old_masks_meta
                agent_kwargs['new_masks_meta'] = new_masks_meta
                agent_kwargs['report_save_dir'] = kwargs['cfg_yl'].agent.save_result_dir
                agent_kwargs['agent_train_loader'] = agent_train_loader
                [agent_loss_iter, reward_step, reward_done] = \
                    agent_business(kwargs['cfg_yl'], kwargs['agent'], kwargs['max_nb_interactions'],
                                   vos_kwargs['n_interaction'], **agent_kwargs)

                reward_step_acc += reward_step
                reward_done_acc += reward_done
                agent_time.update(time.time() - agent_tic)

                # 6 ------ print logs ------
                interaction_time.update(time.time() - interaction_tic)

                _log.info(
                    f"avg_{metric_to_optimize}: {(sum(new_masks_metric) / len(new_masks_metric) * 100):.2f} "
                    f"init_time:{init_time:.2f} "
                    f"rec_time:{frame_recommend_time.val:.2f} "
                    f"seg_time:{segment_time.val:.2f} ({segment_time.avg:.2f})\t"
                    f"next_frame: {next_frame:2d} [{int(sum(new_masks_metric < new_masks_metric[next_frame])) + 1:2d}/{new_masks_metric.shape[0]:2d}]\t"
                    f"reward_step:{reward_step:.2f}  \t"
                    f"reward_done:{reward_done:.2f}  \t"
                    f"seq: {sequence}_{seen_seq[sequence]:1d} [{vos_kwargs['n_interaction']:2d}/{kwargs['max_nb_interactions']:2d}]\t"
                )

                if vos_kwargs['n_interaction'] == kwargs['max_nb_interactions']:
                    final_mask_iou_seq_obj_scb.update(
                        (sum(new_masks_metric) / len(new_masks_metric)) * 100)
                    final_time_seq_obj_scb.update(interaction_time.avg)
                    final_recommend_time_seq_obj_scb.update(
                        frame_recommend_time.avg)
                    final_agent_time_seq_obj_scb.update(agent_time.avg)
                    final_seg_time_seq_obj_scb.update(segment_time.avg)
                    final_reward_step_seq_obj_scb.update(reward_step_acc)
                    final_reward_done_seq_obj_scb.update(reward_done_acc)
                    if agent_loss_iter > 0:
                        agent_loss.update(agent_loss_iter)
                    _log.info(
                        f"* avg_time: {final_time_seq_obj_scb.val:.2f} ({final_time_seq_obj_scb.avg:.2f})"
                        f" rec_time:{final_recommend_time_seq_obj_scb.val:.2f} ({final_recommend_time_seq_obj_scb.avg:.2f})"
                        f" agent_time:{final_agent_time_seq_obj_scb.val:.2f} ({final_agent_time_seq_obj_scb.avg:.2f})\t"
                        f"seg_time: {final_seg_time_seq_obj_scb.val:.2f} ({final_seg_time_seq_obj_scb.avg:.2f})\t"
                        f"{metric_to_optimize}: {final_mask_iou_seq_obj_scb.val:.2f} ({final_mask_iou_seq_obj_scb.avg:.2f})\t"
                        f"reward_step: {final_reward_step_seq_obj_scb.val:.2f} ({final_reward_step_seq_obj_scb.avg:.2f})\t"
                        f"reward_done: {final_reward_done_seq_obj_scb.val:.2f} ({final_reward_done_seq_obj_scb.avg:.2f})\t"
                        f"agent_loss: {agent_loss.val:.4f} ({agent_loss.avg:.4f})\t"
                        f"seq: [{i_seq}/{len(sess.samples)}] {sequence}_{seen_seq[sequence]:1d}"
                    )

            # save agent checkpoint
            save_agent_checkpoint(kwargs['agent'].policy_net, ckpt_dir=kwargs['cfg_yl'].ckpt_dir)

            global_summary = sess.get_global_summary()
            _log.info(f"# final avg {metric_to_optimize}: {final_mask_iou_seq_obj_scb.avg:.4f}\t"
                      f"final agent loss: {agent_loss.avg:.2f}\t"
                      f"final avg reward_step: {final_reward_step_seq_obj_scb.avg:.4f}\t"
                      f"final avg reward_done: {final_reward_done_seq_obj_scb.avg:.4f}")

            auc_round = np.trapz(global_summary['curve'][metric_to_optimize][:-1]) / \
                        (len(global_summary['curve'][metric_to_optimize][:-1]) - 1)
            global_summary['auc'] = auc_round
            auc = float(global_summary['auc'])
            metric_at_threshold = float(
                global_summary['metric_at_threshold'][metric_to_optimize])
            auc_meter.update(auc)
            metric_at_threshold_meter.update(metric_at_threshold)
            _log.info(f"# global_summary: auc:{auc:.4f} ({auc_meter.avg:.4f})\t"
                      f"auc_t:{auc*100:.4f}\t"
                      f"{metric_to_optimize}@{int(global_summary['metric_at_threshold']['threshold']):2d}: "
                      f"{metric_at_threshold:.4f} ({metric_at_threshold_meter.avg:.4f})")
            print(f"# time:\t", end=' ')
            for i in range(len(global_summary['curve']['time'])):
                print(f"{global_summary['curve']['time'][i]:.2f}\t", end=' ')
            print(f"\n# {metric_to_optimize}:\t", end=' ')
            for i in range(len(global_summary['curve'][metric_to_optimize])):
                print(f"{global_summary['curve'][metric_to_optimize][i] * 100:.2f}\t", end=' ')
            print('\n')
