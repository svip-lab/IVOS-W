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
from torchvision import transforms
from torch.utils.data import DataLoader

from sacred import Experiment
from easydict import EasyDict as edict

from davisinteractive.dataset import Davis
from davisinteractive.session import DavisInteractiveSession
from davisinteractive import utils as interactive_utils
from davisinteractive.utils.scribbles import scribbles2mask

sys.path.append(os.path.join('utils', 'config_manet'))
sys.path.append(os.path.join('VOS', 'MANet'))
from utils.misc import (set_random_seed, load_agent_checkpoint, load_network_checkpoint, AverageMeter, sequence_metric)
from utils.utils_agent import recommend_frame
from models.agent import Agent
from models.assessment import AssessNet

from config import cfg
from utils.utils_manet import (load_network, rough_ROI, preprocess, get_results)
from dataloaders.davis_2017_f import DAVIS2017_Feature_Extract
import dataloaders.custom_transforms_f as tr
from networks.deeplab import DeepLab
from networks.IntVOS import IntVOS

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


    # ------ MANet ------
    total_frame_num_dic={}
    #################
    seqs = []
    with open(os.path.join(dataset_root_dir, 'ImageSets', '2017', subset + '.txt')) as f:
        seqs_tmp = f.readlines()
        seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
        seqs.extend(seqs_tmp)
    h_w_dic={}
    for seq_name in seqs:
        images = np.sort(os.listdir(os.path.join(dataset_root_dir, 'JPEGImages/480p/', seq_name.strip())))
        total_frame_num_dic[seq_name]=len(images)
        im_ = cv2.imread(os.path.join(dataset_root_dir, 'JPEGImages/480p/',seq_name,'00000.jpg'))
        im_ = np.array(im_,dtype=np.float32)
        hh_,ww_ = im_.shape[:2]
        h_w_dic[seq_name]=(hh_,ww_)
    ##################
    seq_imgnum_dict_={}
    seq_imgnum_dict=os.path.join(dataset_root_dir,'ImageSets','2017',subset+'_imgnum.txt')
    if os.path.isfile(seq_imgnum_dict):

        seq_imgnum_dict_=json.load(open(seq_imgnum_dict,'r'))
    else:
        for seq in os.listdir(os.path.join(dataset_root_dir,'JPEGImages/480p/')):
            seq_imgnum_dict_[seq]=len(os.listdir(os.path.join(dataset_root_dir,'JPEGImages/480p/',seq)))
        with open(seq_imgnum_dict,'w') as f:
            json.dump(seq_imgnum_dict_,f)

    d = vars(cfg)
    d['TEST_MODE'] = True
    feature_extracter = DeepLab(backbone='resnet',freeze_bn=False)
    with torch.no_grad():
        model = IntVOS(cfg, feature_extracter)
    model= model.cuda()
    print('model loading with default weight...')

    saved_model_dict = os.path.join('VOS', 'MANet', 'save_step_80000.pth')
    pretrained_dict = torch.load(saved_model_dict)
    load_network(model,pretrained_dict)

    print('model loading finished!')
    model.eval()
    resized_h,resized_w = 480,854
    ###############################
    composed_transforms =  transforms.Compose([tr.Resize((resized_h,resized_w)), tr.ToTensor()])
    total_frame_num_dic={}
    #################
    seqs = []
    with open(os.path.join(dataset_root_dir, 'ImageSets', '2017', subset + '.txt')) as f:
        seqs_tmp = f.readlines()
        seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
        seqs.extend(seqs_tmp)
    h_w_dic={}
    for seq_name in seqs:
        images = np.sort(os.listdir(os.path.join(dataset_root_dir, 'JPEGImages/480p/', seq_name.strip())))
        total_frame_num_dic[seq_name]=len(images)
        im_ = cv2.imread(os.path.join(dataset_root_dir, 'JPEGImages/480p/',seq_name,'00000.jpg'))
        im_ = np.array(im_,dtype=np.float32)
        hh_,ww_ = im_.shape[:2]
        h_w_dic[seq_name]=(hh_,ww_)
    _seq_list_file=os.path.join(dataset_root_dir, 'ImageSets', '2017', subset + '_instances.txt')
    if os.path.isfile(_seq_list_file):
        seq_dict = json.load(open(_seq_list_file, 'r'))
    else:
        seq_dict = preprocess(dataset_root_dir, seqs, _seq_list_file)
    ##################
    seq_imgnum_dict_={}
    seq_imgnum_dict=os.path.join(dataset_root_dir, 'ImageSets','2017', subset+'_imgnum.txt')
    if os.path.isfile(seq_imgnum_dict):

        seq_imgnum_dict_=json.load(open(seq_imgnum_dict,'r'))
    else:
        for seq in os.listdir(os.path.join(dataset_root_dir,'JPEGImages/480p/')):
            seq_imgnum_dict_[seq]=len(os.listdir(os.path.join(dataset_root_dir,'JPEGImages/480p/',seq)))
        with open(seq_imgnum_dict,'w') as f:
            json.dump(seq_imgnum_dict_,f)



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

    report_save_dir = os.path.join('results', 'MANet', cfg_yl.setting, cfg_yl.dataset, cfg_yl.method)
    os.makedirs(report_save_dir, exist_ok=True)

    kwargs['model'] = model
    kwargs['seq_dict'] = seq_dict
    kwargs['h_w_dic'] = h_w_dic
    kwargs['composed_transforms'] = composed_transforms

    kwargs['cfg_yl'] = cfg_yl
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

        while sess.next():

            # 1 ------ interaction initial ------
            interaction_tic = time.time()
            init_tic = time.time()
            sequence, scribbles, first_scribble = sess.get_scribbles(only_last=True)
            h,w = kwargs['h_w_dic'][sequence]
            annotated_frames = interactive_utils.scribbles.annotated_frames(scribbles)

            if first_scribble:
                n_objects = Davis.dataset[sequence]['num_objects']
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
                prev_frames = None if kwargs['cfg_yl'].davis_interactive.allow_repeat > 0 else [next_frame]
                annotated_frames_list = [next_frame]
                if kwargs['cfg_yl'].setting == 'wild' and \
                        (kwargs['cfg_yl'].method == 'ours' or kwargs['cfg_yl'].method == 'worst'):
                    mask_quality_pred = np.zeros((n_frame))
                else:
                    mask_quality_pred = None

                # MANet
                embedding_memory = []
                dataset = DAVIS2017_Feature_Extract(root=kwargs['dataset_root_dir'], split=kwargs['subset'],
                                                    transform=kwargs['composed_transforms'],
                                                    seq_name=sequence)
                testloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
                h, w = dataset[0]['meta']['h_w']
                for ii, sample in enumerate(testloader):
                    imgs = sample['img1']
                    imgs = imgs.cuda()
                    with torch.no_grad():
                        frame_embedding = kwargs['model'].extract_feature(imgs)
                    embedding_memory.append(frame_embedding)
                del frame_embedding
                torch.cuda.empty_cache()

                eval_global_map_tmp_dic = {}
                local_map_dics = ({}, {})
                obj_nums = kwargs['seq_dict'][sequence][-1]

                embedding_memory = torch.cat(embedding_memory, 0)
                _, _, emb_h, emb_w = embedding_memory.size()

                prev_label = None
                prev_label_storage = torch.zeros(n_frame, h, w)
                prev_label_storage = prev_label_storage.cuda()

                # VOS
                vos_kwargs = dict()
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

                prev_label = prev_label_storage[next_frame]
                prev_label = prev_label.unsqueeze(0).unsqueeze(0)


            ref_frame_embedding = embedding_memory[next_frame]
            ref_frame_embedding = ref_frame_embedding.unsqueeze(0)
            if subseq is not None:
                scribbles_subseq = [scribbles['scribbles'][i] for i in subseq]
                scribbles['scribbles'] = scribbles_subseq
            scribble_masks = scribbles2mask(scribbles, (emb_h, emb_w))
            scribble_label = scribble_masks[next_frame]
            scribble_sample = {'scribble_label': scribble_label}
            scribble_sample = tr.ToTensor()(scribble_sample)
            scribble_label = scribble_sample['scribble_label']
            scribble_label = scribble_label.unsqueeze(0)
            scribble_label = rough_ROI(scribble_label.cuda()) if first_scribble else scribble_label.cuda()

            scribbles['annotated_frame'] = next_frame
            init_time = time.time() - init_tic

            # 2 ------ segmentation ------
            if annotated_frames:
                segment_tic = time.time()
                while True:
                    try:
                        with torch.no_grad():

                            final_masks, all_P = get_results(kwargs['model'], ref_frame_embedding, scribble_label, prev_label,
                                                             eval_global_map_tmp_dic, local_map_dics, vos_kwargs['n_interaction'],
                                                             sequence, obj_nums, next_frame, first_scribble, h, w,
                                                             prev_label_storage, n_frame, embedding_memory)
                        break
                    except RuntimeError as exception:
                        if "out of memory" in str(exception):
                            _log.info("WARNING: out of memory (vos)")
                            torch.cuda.empty_cache()
                        else:
                            raise exception
                new_masks = final_masks.cpu().numpy()
                new_masks_metric = sequence_metric(metric_to_optimize, gt_masks, new_masks, nb_objects)
                segment_time.update(time.time() - segment_tic)
            else:
                segment_time.update(0)


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
