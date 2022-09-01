# IVOS-W

## Paper

**Learning to Recommend Frame for Interactive Video Object Segmentation in the Wild**

[Zhaoyun Yin](https://github.com/zyy-cn),
[Jia Zheng](http://bertjiazheng.github.io),
[Weixin Luo](https://zachluo.github.io),
[Shenhan Qian](https://shenhanqian.com/),
[Hanling Zhang](http://design.hnu.edu.cn/info/1023/5767.htm),
[Shenghua Gao](https://sist.shanghaitech.edu.cn/sist_en/2020/0814/c7582a54772/page.htm).

IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[[arXiv](https://arxiv.org/abs/2103.10391)]
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_Learning_To_Recommend_Frame_for_Interactive_Video_Object_Segmentation_in_CVPR_2021_paper.pdf)]
[[Supp. Material](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Yin_Learning_To_Recommend_CVPR_2021_supplemental.pdf)]

## Getting Started

Create the environment

```bash
# create conda env
conda create -n ivosw python=3.7
# activate conda env
conda activate ivosw
# install pytorch
conda install pytorch=1.3 torchvision
# install other dependencies
pip install -r requirements.txt
```

We adopt [MANet](https://github.com/lightas/CVPR2020_MANet), [IPN](https://github.com/zyy-cn/IPN.git), and [ATNet](https://github.com/yuk6heo/IVOS-ATNet) as the VOS algorithms. Please follow the instructions to install the dependencies.

```bash
git clone https://github.com/yuk6heo/IVOS-ATNet.git VOS/ATNet
git clone https://github.com/lightas/CVPR2020_MANet.git VOS/MANet
git clone https://github.com/zyy-cn/IPN.git VOS/IPN
```

## Dataset Preparation

- DAVIS 2017 Dataset
  - Download the data and human annotated scribbles [here](https://davischallenge.org/davis2017/code.html).
  - Place `DAVIS` folder into `root/data`.
- YouTube-VOS Dataset
  - Download the YouTube-VOS 2018 version [here](https://youtube-vos.org/dataset).
  - Clean up the annotations following [here](https://competitions.codalab.org/forums/16267/2626/).
  - Download our annotated scribbles [here](https://drive.google.com/file/d/1yliwTYP_PkiJnIAOo292gx9Fv3sLVYj4/view?usp=sharing).

Create a DAVIS-like structure of YouTube-VOS by running the following commands:

```bash
python datasets/prepare_ytbvos.py --src path/to/youtube_vos --scb path/to/scribble_dir
```

## Evaluation

For evaluation, please download the pretrained [agent model](https://drive.google.com/file/d/18OgPfPcYipe_1Ka7qlKar7mVVwvG7gPT/view?usp=sharing) and [quality assessment model](https://drive.google.com/file/d/1Xdkr6Epm5H5hDkQoBqmp0E_t5V-9SlKl/view?usp=sharing), then place them into `root/weights` and run the following commands:

```bash
python eval_agent_{atnet/manet/ipn}.py with setting={oracle/wild} dataset={davis/ytbvos} method={random/linspace/worst/ours}
```

The results will be stored in `results/{VOS}/{setting}/{dataset}/{method}/summary.json`

**Note**: The results may fluctuate slightly with different versions of *networkx*, which is used by davisinteractive to generate simulated scribbles.

## Training

First, prepare the data used to train the agent by downloading [reward records](https://drive.google.com/file/d/1cNIstWStaGCknoAkBUquYEpYwz0iFmUn/view?usp=sharing) and [pretrained experience buffer](https://drive.google.com/file/d/13rXLrWSiXdhk5XB3jyiVZHh7mbeVX8p_/view?usp=sharing), place them into `root/train`, or generate them from scratch:

```bash
python produce_reward.py
python pretrain_agent.py
```

To train the agent:

```bash
python train_agent.py
```

To train the segmentation quality assessment model:

```bash
python generate_data.py
python quality_assessment.py
```

## Citation

```bibtex
@inproceedings{IVOSW,
  title     = {Learning to Recommend Frame for Interactive Video Object Segmentation in the Wild},
  author    = {Zhaoyuan Yin and
               Jia Zheng and
               Weixin Luo and
               Shenhan Qian and
               Hanling Zhang and
               Shenghua Gao},
  booktitle = {CVPR},
  year      = {2021}
}
```

## LICENSE

The code is released under the [MIT license](LICENSE).
