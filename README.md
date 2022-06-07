# Generic Boundary Event Captioning Challenge at CVPR 2022 LOVEU workshop

*Jaehyuk Heo, YongGi Jeong, Sunwoo Kim, Jaehee Kim, Pilsung Kang*  
*School of Industrial & Management Engineering, Korea University*  
*Seoul, Korea*


We proposed the Rich Encoder-decoder framework for Video Event Captioner (REVECA). Our model achieve 3rd place in GEBC Challenge [ [link](https://codalab.lisn.upsaclay.fr/competitions/4157#results) ].

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

1. We use the semantic segmentation mask for training model. The segmentation model is [Mask2Former](https://github.com/facebookresearch/Mask2Former).

2. We use TSN features extracted by Temporal Segmentation Networks. TSN features released in GEBC Challenge can download in [here](https://drive.google.com/drive/folders/1kOauKJY4MphWJhjYcXcCcdmP-071Fu6D?usp=sharing).


# Methods

Our video understanding model called REVECA based on CoCa. We use three methods: (1) Temporal-based Pairwise Difference (TPD), (2) Frame position embedding and (3) LoRA. we use timm version == 0.6.2.dev0 and `loralib`. And then, we modify `a vision_transformer.py` for using LoRA. 

<p align='center'>
    <img width='800' src=''>
</p>


# Results

Method | Avg. | CIDEr | SPICE | ROUGE-L
---|---|---|---|---
CNN+LSTM | 29.94 | 49.73 | 13.62 | 26.46
Robust Change Captioning | 34.16 | 58.56 | 16.34 | 27.57
UniVL-revised | 36.64 | 65.74 | 18.06 | 26.12
ActBERT-revised | 40.80 | 74.71 | 19.52 | 28.15
**REVECA (our model)** | **50.97** | **93.91** | **24.66** | **34.34**

# Saved Model

Our final model weights can download in [here]().


# Citation
