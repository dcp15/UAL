# Reliability-Aware Prediction via Uncertainty Learning for Person Image Retrieval



## introduction

This repository is the code for ECCV2022 paper: Reliability-Aware Prediction via Uncertainty Learning for Person Image Retrieval. 

The code is based on the repository [fast-reid](https://github.com/JDAI-CV/fast-reid). Please refer to it for details.



## Training & Evaluation

To train a model, first setup the corresponding datasets following [datasets/README.md](https://github.com/JDAI-CV/fast-reid/tree/master/datasets), then run:

```python'''
python3 tools/train_net.py --config-file ./configs./configs/MSMT17/bagtricks_R50_bayes.yml MODEL.DEVICE "cuda:0"
```

To evaluate a model's performance, use

```python"""
python3 tools/train_net.py --config-file ./configs/Market1501/bagtricks_R50_bayes.yml --eval-only \
MODEL.WEIGHTS /path/to/checkpoint_file MODEL.DEVICE "cuda:0"
```



## Results



MSMT17 baselines:

|           | BOT(R50) |          | BOT(R50-ibn) |          | BOT(S50) |          |
| :-------: | :------: | :------: | :----------: | -------- | :------: | :------: |
|           |    R1    |   mAP    |      R1      | mAP      |    R1    |   mAP    |
| fast-reid |   73.9   |   49.9   |     79.1     | 55.4     |   81.0   |   59.4   |
| **Ours**  | **78.7** | **53.6** |   **82.4**   | **59.1** | **84.7** | **65.3** |

