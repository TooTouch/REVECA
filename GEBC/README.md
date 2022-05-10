# GEBC Baseline

- revised_ActBERT


# Environments

- `../docker/Dockerfile` 
- requirements.txt
- 

```{base}
pip install -r requirements.txt
```

Install `pycocoevalcap` for evaluation

```{base}
git clone https://github.com/LuoweiZhou/coco-caption.git

cd coco-caption
bash ./get_stanford_models.sh

mv pycocoevalcap ../utils/

cd ..
rm -r coco-caption
```


# Run

```{base}
bash run.sh
```