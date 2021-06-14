import os
import json
import shutil
import argparse
import numpy as np
from PIL import Image

def getSeqInfo(dataset_dir, seq):

    ann_dir = os.path.join(dataset_dir, 'Annotations', '480p')
    seq_path = os.path.join(ann_dir, seq)
    frame_list = os.listdir(seq_path)
    frame_num = len(frame_list)

    frames = os.listdir(os.path.join(ann_dir, seq))
    masks = np.stack([np.array(Image.open(os.path.join(ann_dir, seq, f)).convert('P'), dtype=np.uint8) for f in frames])
    img_size = [masks.shape[1], masks.shape[0]]
    obj_ids = np.delete(np.unique(masks), 0)

    return frame_num, img_size, len(obj_ids)


def create_json(root_dir):
    val_txt_dst = os.path.join(root_dir, 'ImageSets', '2017', 'val.txt')

    with open(val_txt_dst, 'r') as f:
        val_seqs = f.readlines()
    f.close()
    val_seqs = list(map(lambda elem: elem.strip(), val_seqs))

    # create davis.json
    '''Generate global json'''

    json_dict = dict()
    json_dict['attributes'] = []
    json_dict['sets'] = ["train", "val"]
    json_dict['years'] = [2018]
    json_dict['sequences'] = dict()

    for idx, seq in enumerate(val_seqs):
        seq = seq.strip()
        seq_dict = {'attributes': [], 'eval_t': True, 'name': seq, 'set': 'val', 'year': 2018, 'num_scribbles': 3}

        seq_dict['num_frames'], seq_dict['image_size'], seq_dict['num_objects'] = getSeqInfo(root_dir, seq)

        json_dict['sequences'][seq] = seq_dict
        print(f'valid: {idx+1}')

    global_json_path = os.path.join(root_dir, 'scb_ytbvos.json')
    with open(global_json_path, 'wt') as f:
        json.dump(json_dict, f, indent=2, separators=(',', ': '))


def create_dataset(src_ytbvos_path, dst_ytbvos_path, scb_ytbvos_path):

    if os.path.exists(src_ytbvos_path):
        os.makedirs(dst_ytbvos_path, exist_ok=True)

        # set youtube original path
        src_dir_JPEGImages = os.path.join(src_ytbvos_path, 'train', 'JPEGImages')
        src_dir_Annotations = os.path.join(src_ytbvos_path, 'train', 'CleanedAnnotations')

        # set youtube davis-like path
        dst_dir_ImageSets = os.path.join(dst_ytbvos_path, 'ImageSets', '2017')
        dst_dir_JPEGImages = os.path.join(dst_ytbvos_path, 'JPEGImages', '480p')
        dst_dir_Annotations = os.path.join(dst_ytbvos_path, 'Annotations', '480p')
        dst_dir_Scribbles = os.path.join(dst_ytbvos_path, 'Scribbles')

        if os.path.isdir(src_dir_JPEGImages) and os.path.isdir(src_dir_Annotations) and os.path.isdir(scb_ytbvos_path):
            # load sequence list
            assert len(os.listdir(src_dir_JPEGImages)) == len(os.listdir(src_dir_Annotations))
            with open(os.path.join(scb_ytbvos_path, 'val.txt'), 'r') as f:
                seqs_list = f.readlines()
            f.close()
            seqs_list = list(map(lambda elem: elem.strip(), seqs_list))

        else:
            if not os.path.isdir(src_dir_JPEGImages): print(f"{src_dir_JPEGImages} is not found in {src_ytbvos_path}")
            if not os.path.isdir(src_dir_Annotations): print(f"{src_dir_Annotations} is not found in {src_ytbvos_path}")
            if not os.path.isdir(scb_ytbvos_path): print(f"{scb_ytbvos_path} is not found")
            return

        # create dist dirs
        os.makedirs(dst_dir_ImageSets, exist_ok=True)
        os.makedirs(dst_dir_JPEGImages, exist_ok=True)
        os.makedirs(dst_dir_Annotations, exist_ok=True)
        os.makedirs(dst_dir_Scribbles, exist_ok=True)

        # --- copy files ---
        # ImageSets
        shutil.copyfile(os.path.join(scb_ytbvos_path, 'val.txt'), os.path.join(dst_dir_ImageSets, 'val.txt'))

        len_seq = []
        for i, seq in enumerate(seqs_list):
            print(f"validation set {i+1}")
            # JPEGImages
            src_dir_JPEGImages_seq = os.path.join(src_dir_JPEGImages, seq)
            dst_dir_JPEGImages_seq = os.path.join(dst_dir_JPEGImages, seq)
            os.makedirs(dst_dir_JPEGImages_seq, exist_ok=True)
            file_name = np.sort(os.listdir(src_dir_JPEGImages_seq))
            for j, file in enumerate(file_name):
                src_path = os.path.join(src_dir_JPEGImages_seq, file)
                dst_path = os.path.join(dst_dir_JPEGImages_seq, f"{str(j).zfill(5)}.jpg")
                if not os.path.exists(dst_path): shutil.copyfile(src_path, dst_path)
                # if not os.path.exists(dst_path): os.symlink(src_path, dst_path)

            # Annotations
            src_dir_Annotations_seq = os.path.join(src_dir_Annotations, seq)
            dst_dir_Annotations_seq = os.path.join(dst_dir_Annotations, seq)
            os.makedirs(dst_dir_Annotations_seq, exist_ok=True)
            file_name = np.sort(os.listdir(src_dir_Annotations_seq))
            for j, file in enumerate(file_name):
                src_path = os.path.join(src_dir_Annotations_seq, file)
                dst_path = os.path.join(dst_dir_Annotations_seq, f"{str(j).zfill(5)}.png")
                if not os.path.exists(dst_path): shutil.copyfile(src_path, dst_path)
                # if not os.path.exists(dst_path): os.symlink(src_path, dst_path)

            # Scribbles
            src_dir_Scribbles_seq = os.path.join(scb_ytbvos_path, seq)
            dst_dir_Scribbles_seq = os.path.join(dst_dir_Scribbles, seq)
            os.makedirs(dst_dir_Scribbles_seq, exist_ok=True)
            file_name = np.sort(os.listdir(src_dir_Scribbles_seq))
            for j, file in enumerate(file_name):
                src_path = os.path.join(src_dir_Scribbles_seq, file)
                dst_path = os.path.join(dst_dir_Scribbles_seq, file)
                if not os.path.exists(dst_path): shutil.copyfile(src_path, dst_path)

            # statistic
            file_name = np.sort(os.listdir(src_dir_JPEGImages_seq))
            len_seq.append(len(file_name))

        # create sequences information
        create_json(dst_ytbvos_path)

        print(f"done")
    else:
        print(f"{src_ytbvos_path} not existed")



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--scb', type=str, required=True)
    parser.add_argument('--dst', type=str, default='data/Scribble_Youtube_VOS')
    args = parser.parse_args()

    src_ytbvos_path = args.src
    dst_ytbvos_path = args.dst
    scb_ytbvos_path = args.scb

    create_dataset(src_ytbvos_path, dst_ytbvos_path, scb_ytbvos_path)


if __name__ == '__main__':
    main()