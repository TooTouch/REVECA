# Generic Boundary Event Captioning Challenge at CVPR 2022 LOVEU workshop

*Jaehyuk Heo, YongGi Jeong, Sunwoo Kim, Jaehee Kim, Pilsung Kang*  
*School of Industrial & Management Engineering, Korea University*  
*Seoul, Korea*


We propose the Rich Encoder-decoder framework for Video Event Captioner (REVECA). Our model achieves 3rd place in [GEBC Challenge](https://codalab.lisn.upsaclay.fr/competitions/4157#results).

<p align='center'>
    <img width='800' src='https://github.com/TooTouch/REVECA/blob/main/assets/figure1.png'>
</p>

# Environments

1. Build a docker image and make a docker container

```bash
cd docker 
bash docker_build.sh $image_name
```

2. Install packages

```bash
pip install -r requirements
```

# Datasets

Download Kinetics-GEBC and annotations in [here](https://sites.google.com/view/loveucvpr22/home?authuser=0). And save files in `./datasets`

```
datasets/
└── annotations
    ├── testset_highest_f1.json
    ├── trainset_highest_f1.json
    ├── valset_highest_f1.json
```


Our model uses three video features: instance segmentation mask, TSN features

1. We use the semantic segmentation mask for the training model. The segmentation model is [Mask2Former](https://github.com/facebookresearch/Mask2Former).

![](https://github.com/TooTouch/REVECA/blob/main/assets/run_with_seg.gif)

2. We use TSN features extracted by Temporal Segmentation Networks. TSN features released in GEBC Challenge can download [here](https://drive.google.com/drive/folders/1kOauKJY4MphWJhjYcXcCcdmP-071Fu6D?usp=sharing).


# Methods

Our video understanding model is called REVECA, based on CoCa. We use three methods: (1) Temporal-based Pairwise Difference (TPD), (2) Frame position embedding, and (3) LoRA. we use timm version == 0.6.2.dev0 and `loralib`. And then, we modify a `vision_transformer.py` for using LoRA. 


# Results

Method | Avg. | CIDEr | SPICE | ROUGE-L
---|---|---|---|---
CNN+LSTM | 29.94 | 49.73 | 13.62 | 26.46
Robust Change Captioning | 34.16 | 58.56 | 16.34 | 27.57
UniVL-revised | 36.64 | 65.74 | 18.06 | 26.12
ActBERT-revised | 40.80 | 74.71 | 19.52 | 28.15
**REVECA (our model)** | **50.97** | **93.91** | **24.66** | **34.34**

# Saved Model

Our final model weights can download [here](https://drive.google.com/file/d/1sQZXg5-L6i5l6brCyu5HCsaoRvlVSiuO/view?usp=sharing).


# Citation
