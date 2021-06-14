import os
import sys
import cv2
import copy
import time
import json
import logging
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from sacred import Experiment
from easydict import EasyDict as edict

from davisinteractive.dataset import Davis
from davisinteractive.session import DavisInteractiveSession
from davisinteractive import utils as interactive_utils

sys.path.append(os.path.join('VOS', 'ATNet'))
from utils.misc import (set_random_seed, load_agent_checkpoint, load_network_checkpoint, AverageMeter, sequence_metric)
from utils.utils_agent import recommend_frame
from utils.utils_atnet import run_VOS_singleiact
from models.agent import Agent
from models.assessment import AssessNet

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


ex = Experiment('eval')
ex.logger = create_basic_stream_logger('%(name)s - %(message)s')
ex.add_config('./configs/config.yaml')


def davis_config(run, _log):

    # ------ configs ------
    kwargs = dict()
    cfg_yl = edict(run.config)
    cfg_yl.phase = 'eval'

    device = torch.device(f"cuda:{cfg_yl.gpu_id}" if torch.cuda.is_available() else "cpu")
    subset = 'val'
    max_nb_interactions = 8
    max_time = None  # Maximum time per object

    set_random_seed(cfg_yl.seed)

    if cfg_yl.dataset == 'davis':
        dataset_root_dir = cfg_yl.data.root_dir_davis
    elif cfg_yl.dataset == 'ytbvos':
        dataset_root_dir = cfg_yl.data.root_dir_scribble_youtube_vos
        from davisinteractive.dataset.davis import _SETS
        from davisinteractive.dataset.davis import _DATASET
        _SETS['train'], _SETS['val'], _SETS['trainval'] = [], [], []
        _DATASET['sequences'].clear()
        with open(os.path.join(dataset_root_dir, 'scb_ytbvos.json')) as fp:
            DATASET = json.load(fp)
        for k, v in DATASET['sequences'].items():
            _DATASET['sequences'][k] = v
        for s in _DATASET['sequences'].values():
            _SETS[s['set']].append(s['name'])
        _SETS['trainval'] = _SETS['train'] + _SETS['val']
    else:
        raise NotImplementedError

    davis = Davis(davis_root=dataset_root_dir)


    # ------ ATNet ------
    config = Config()
    config.davis_dataset_dir = dataset_root_dir
    net = ATnet()
    net.cuda()
    net.eval()
    net.load_state_dict(torch.load(os.path.join('VOS', 'ATNet', config.test_load_state_dir)))


    if cfg_yl.method == 'ours':
        # ------ Agent ------
        agent = Agent(device=device, cfg=cfg_yl)
        if load_agent_checkpoint(agent, cfg_yl.ckpt_dir, device=device, strict=True):
            print(f"success load agent ckpt")
        else:
            print(f"fail to load agent ckpt")

        # ------ Assess_net ------
        if cfg_yl.setting == 'oracle':
            assess_net = None
            print(f"assess_net is unavailable")
        elif cfg_yl.setting == 'wild':
            assess_net = AssessNet()
            assess_net_dir = os.path.join(cfg_yl.ckpt_dir, 'assess_net.pt')
            if load_network_checkpoint(assess_net_dir, assess_net, device='cpu'):
                print(f"success load assess_net ckpt from {assess_net_dir}")
            else:
                print(f"fail to load assess_net ckpt")
            assess_net.to(device)
            assess_net.eval()
        else:
            raise NotImplementedError
    elif cfg_yl.method == 'worst':
        agent = None
        cfg_yl.davis_interactive.allow_repeat = 0

        # ------ Assess_net ------
        if cfg_yl.setting == 'oracle':
            assess_net = None
            print(f"assess_net is unavailable")
        elif cfg_yl.setting == 'wild':
            assess_net = AssessNet()
            assess_net_dir = os.path.join(cfg_yl.ckpt_dir, 'assess_net.pt')
            if load_network_checkpoint(assess_net_dir, assess_net, device='cpu'):
                print(f"success load assess_net ckpt from {assess_net_dir}")
            else:
                print(f"fail to load assess_net ckpt")
            assess_net.to(device)
            assess_net.eval()
        else:
            raise NotImplementedError
    elif cfg_yl.method == 'random':
        assert cfg_yl.setting == 'wild'
        agent = None
        assess_net = None
    elif cfg_yl.method == 'linspace':
        assert cfg_yl.setting == 'wild'
        agent = None
        assess_net = None
        cfg_yl.davis_interactive.allow_repeat = 0
    else:
        raise NotImplementedError

    report_save_dir = os.path.join('results', 'ATNet', cfg_yl.setting, cfg_yl.dataset, cfg_yl.method)
    os.makedirs(report_save_dir, exist_ok=True)

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

    return kwargs


@ex.automain
def main(_run, _log):

    kwargs = davis_config(_run, _log)

    seen_seq = {}

    # 'J', 'F', 'J_AND_F'
    metric_to_optimize = kwargs['cfg_yl'].davis_interactive.metric

    with DavisInteractiveSession(
        host='localhost', davis_root=kwargs['dataset_root_dir'], subset=kwargs['subset'],
        metric_to_optimize=metric_to_optimize, max_nb_interactions=kwargs['max_nb_interactions'],
        max_time=kwargs['max_time'], report_save_dir=kwargs['report_save_dir']) as sess:

        # per object per serquence
        final_mask_quality_seq_obj_scb = AverageMeter()
        final_time_seq_obj_scb = AverageMeter()
        final_recommend_time_seq_obj_scb = AverageMeter()
        final_seg_time_seq_obj_scb = AverageMeter()
        corr_meter_seq_obj_scb = AverageMeter()
        diff_meter_seq_obj_scb = AverageMeter()
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
                corr_meter = AverageMeter()
                diff_meter = AverageMeter()

                gt_masks = kwargs['davis'].load_annotations(sequence)
                nb_objects = kwargs['davis'].dataset[sequence]['num_objects']

                assert len(annotated_frames) > 0
                next_frame = annotated_frames[0]
                first_frame = annotated_frames[0]
                if sequence not in seen_seq.keys():
                    seen_seq[sequence] = 1
                    jpeg_dir_path = os.path.join(kwargs['dataset_root_dir'], 'JPEGImages', '480p', sequence)
                    all_F = torch.Tensor((np.stack([np.array(cv2.imread(os.path.join(jpeg_dir_path, frame)),
                                                             dtype=np.float32)[:, :, [2, 1, 0]]/255.
                                                    for frame in np.sort(os.listdir(jpeg_dir_path))
                                                    ], 0).transpose((0, 3, 1, 2))))
                else:
                    seen_seq[sequence] += 1
                # make subsequence information
                n_frame = len(scribbles['scribbles'])
                subseq = None
                prev_frames = None if kwargs['cfg_yl'].davis_interactive.allow_repeat > 0 else [
                    next_frame]
                annotated_frames_list = [next_frame]
                if kwargs['cfg_yl'].setting == 'wild' and \
                        (kwargs['cfg_yl'].method == 'ours' or kwargs['cfg_yl'].method == 'worst'):
                    mask_quality_pred = np.zeros((n_frame))
                else:
                    mask_quality_pred = None

                # ATNet
                anno_dict = {'frames': [], 'annotated_masks': [],
                             'masks_tobe_modified': []}
                info = Davis.dataset[sequence]
                img_size = info['image_size'][::-1]
                n_objects = info['num_objects']
                final_masks = np.zeros([n_frame, img_size[0], img_size[1]])
                vos_kwargs = dict()
                vos_kwargs['pad_info'] = utils.apply_pad(final_masks[0])[1]
                vos_kwargs['hpad1'], vos_kwargs['wpad1'] = vos_kwargs['pad_info'][0][0], vos_kwargs['pad_info'][1][0]
                vos_kwargs['hpad2'], vos_kwargs['wpad2'] = vos_kwargs['pad_info'][0][1], vos_kwargs['pad_info'][1][1]
                h_ds, w_ds = int((img_size[0] + sum(vos_kwargs['pad_info'][0])) / 4), \
                    int((img_size[1] + sum(vos_kwargs['pad_info'][1])) / 4)
                vos_kwargs['anno_6chEnc_r5_list'], vos_kwargs['anno_3chEnc_r5_list'] = [
                ], []
                vos_kwargs['prob_map_of_frames'] = torch.zeros(
                    (n_frame, n_objects, 4 * h_ds, 4 * w_ds)).cuda()
                vos_kwargs['num_frames'] = n_frame
                vos_kwargs['n_objects'] = n_objects
                vos_kwargs['n_interaction'] = 1
                vos_kwargs['subseq'] = subseq

                rec_kwargs = dict()
                rec_kwargs['n_frame'] = n_frame
                rec_kwargs['n_objects'] = n_objects
                rec_kwargs['all_F'] = all_F
                rec_kwargs['mask_quality'] = mask_quality_pred

            else:
                annotated_frames_list.append(next_frame)
                vos_kwargs['n_interaction'] += 1

            # Where we save annotated frames
            anno_dict['frames'].append(next_frame)
            # mask before modefied at the annotated frame
            anno_dict['masks_tobe_modified'].append(final_masks[next_frame])
            scribbles['annotated_frame'] = next_frame
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

            annotated_frames_list_np = np.zeros(len(new_masks_metric))
            for i in annotated_frames_list:
                annotated_frames_list_np[i] += 1

            rec_kwargs['all_P'] = all_P
            rec_kwargs['new_masks_quality'] = new_masks_metric
            rec_kwargs['prev_frames'] = prev_frames
            rec_kwargs['annotated_frames_list'] = copy.deepcopy(annotated_frames_list)
            rec_kwargs['first_frame'] = first_frame
            rec_kwargs['max_nb_interactions'] = kwargs['max_nb_interactions']
            next_frame = recommend_frame(kwargs['cfg_yl'], kwargs['assess_net'], kwargs['agent'], kwargs['device'],
                                         **rec_kwargs)

            if rec_kwargs['prev_frames'] is not None:
                rec_kwargs['prev_frames'].append(next_frame)
            frame_recommend_time.update(time.time() - frame_recommend_tic)

            # 4 ------ Submit prediction ------
            sess.submit_masks(new_masks, next_scribble_frame_candidates=[next_frame])

            # 5 ------ print logs ------
            corr = np.corrcoef([new_masks_metric, mask_quality_pred])[0, 1] if mask_quality_pred is not None else np.nan
            corr_meter.update(corr)
            diff = F.mse_loss(torch.Tensor(mask_quality_pred), torch.Tensor(new_masks_metric)) \
                if mask_quality_pred is not None else np.nan
            diff_meter.update(diff)
            interaction_time.update(time.time() - interaction_tic)

            _log.info(
                f"avg_{metric_to_optimize}: {(sum(new_masks_metric) / len(new_masks_metric) * 100):.2f} "
                f"init_time:{init_time:.2f} "
                f"rec_time:{frame_recommend_time.val:.2f} "
                f"seg_time:{segment_time.val:.2f} ({segment_time.avg:.2f})\t"
                f"next_frame: {next_frame:2d} [{int(sum(new_masks_metric < new_masks_metric[next_frame])) + 1:2d}/{new_masks_metric.shape[0]:2d}]\t"
                f"corr: {corr:.2f} ({corr_meter.avg:.2f}) ({corr_meter_seq_obj_scb.avg:.2f})\t"
                f"diff: {diff:.2f} ({diff_meter.avg:.2f}) ({diff_meter_seq_obj_scb.avg:.2f})\t"
                f"seq: {sequence}_{seen_seq[sequence]:1d} [{vos_kwargs['n_interaction']:2d}/{kwargs['max_nb_interactions']:2d}]\t"
            )

            if vos_kwargs['n_interaction'] == kwargs['max_nb_interactions']:
                final_mask_quality_seq_obj_scb.update(
                    (sum(new_masks_metric) / len(new_masks_metric)) * 100)
                final_time_seq_obj_scb.update(interaction_time.avg)
                final_recommend_time_seq_obj_scb.update(
                    frame_recommend_time.avg)
                final_seg_time_seq_obj_scb.update(segment_time.avg)
                corr_meter_seq_obj_scb.update(corr_meter.avg)
                diff_meter_seq_obj_scb.update(diff_meter.avg)
                _log.info(
                    f"* avg_time: {final_time_seq_obj_scb.val:.2f} ({final_time_seq_obj_scb.avg:.2f})"
                    f" rec_time:{final_recommend_time_seq_obj_scb.val:.2f} ({final_recommend_time_seq_obj_scb.avg:.2f})"
                    f"seg_time: {final_seg_time_seq_obj_scb.val:.2f} ({final_seg_time_seq_obj_scb.avg:.2f})\t"
                    f"{metric_to_optimize}: {final_mask_quality_seq_obj_scb.val:.2f} ({final_mask_quality_seq_obj_scb.avg:.2f})\t"
                    f"corr: {corr_meter_seq_obj_scb.val:.2f} ({corr_meter_seq_obj_scb.avg:.2f})\t"
                    f"diff: {diff_meter_seq_obj_scb.val:.2f} ({diff_meter_seq_obj_scb.avg:.2f})\t"
                    f"seq: [{i_seq}/{len(sess.samples)}] {sequence}_{seen_seq[sequence]:1d}"
                )

        global_summary = sess.get_global_summary()
        _log.info(f"# final avg {metric_to_optimize}: {final_mask_quality_seq_obj_scb.avg:.4f}\t"
                  f"final avg corr: {corr_meter_seq_obj_scb.avg:.4f}\t"
                  f"final avg diff: {diff_meter_seq_obj_scb.avg:.4f}")

        auc = np.trapz(global_summary['curve'][metric_to_optimize][:-1]) / \
                    (len(global_summary['curve'][metric_to_optimize][:-1]) - 1)
        _log.info(f"# global_summary: auc:{auc*100:.4f}")
        print(f"\n# {metric_to_optimize}:\t", end=' ')
        for i in range(len(global_summary['curve'][metric_to_optimize]) - 1):
            print(f"{global_summary['curve'][metric_to_optimize][i] * 100:.2f}\t", end=' ')
        print('\n')

        summary = {'auc':auc, "curve": {metric_to_optimize: global_summary['curve'][metric_to_optimize][:-1]}}
        with open(os.path.join(kwargs['report_save_dir'], 'summary.json'), 'w') as fp:
            json.dump(summary, fp)
