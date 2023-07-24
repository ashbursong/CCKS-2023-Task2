# Code for CCKS 2023 task2, inductive link prediction

Team Member:
- [Tianyu Liu](tianyu_liu@mail.ustc.edu.cn), 
- [Deqiang Huang](deqianghuang@mail.ustc.edu.cn), 
- [Guoqing Zhao](2208513233@qq.com) 
(All of us are students of USTC.)

## Preparation
1. All the data and pretrained model are included into `data` folder and `exp/NBFNet/CCKS` folder.
2. Preparing conda environment by running these commands:
```shell
# conda install
conda install pytorch=1.8.0 cudatoolkit=11.1 pyg -c pytorch -c pyg -c conda-forge
conda install ninja easydict pyyaml -c conda-forge

# pip install
pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.0.8 torch-sparse==0.6.12 torch-geometric -f https://data.pyg.org/whl/torch-1.8.0+cu111.html
pip install ninja easydict pyyaml
```

## Training
To train a model, just run this command:
```shell
python script/run.py -c config/inductive/ccks.yaml --gpus [0]
```

All the training hyper-parameters are stored in `config/inductive/ccks.yaml`. Feel free to change them to get different results.

## Inference
To inference with the test data and generate submission files, just run this command:

```shell
python script/inference.py -c config/inductive/ccks.yaml --gpus [0]
```

This will generate a `test.json` file in the experiment folder (given by `ccks.yaml`). 

A `scores.pt` file will be output in the experiment folder as well, which represents the prediction of a specific model.

## Reproduce the best results
We perform grid search for best hyper-parameters. Each experiment has a unique config file (in `config/inductive/grid_search`). Then we perform stacking ensemble to get best results.
To reproduce the best results, just run:

```shell
python script/ensemble.py
```

This will generate a `test.json` file in the current path.

## Citation

We use the official codebase of [NBFNet](https://arxiv.org/pdf/2106.06935.pdf), thanks for their contribution.
