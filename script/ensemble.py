import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import ipdb
import numpy as np
import pandas as pd
from nbfnet import tasks, util
import tqdm
import ipdb
torch.manual_seed(1024)


def prepare_batch():
    test_query = pd.read_json("data/CCKS/test_query.json")
    head_rel = test_query.iloc[0]
    tails = test_query.iloc[1]
    
    batch = []
    for i in range(head_rel.shape[0]):
        hr = head_rel[i]
        t_list = tails[i]
        for t in t_list:
            batch.append([hr[0], t, hr[1]])
    batch = torch.tensor(batch)
    return batch, head_rel, tails


if __name__ == "__main__":
    
    grid_search_id = [30,30,30,54, 62]
    grid_search_exp = [f"exp/NBFNet/CCKS/grid_search_{i}/scores.pt" for i in grid_search_id]
    
    scores = []
    for i, path in enumerate(grid_search_exp):
        if not os.path.exists(path):
            os.system(f"python script/inference.py -c config/inductive/grid_search/grid_search_{grid_search_id[i]}.yaml --gpus [6]")
        scores.append(torch.load(path))
    
    scores = torch.stack(scores).mean(dim=0)
    
    batch, head_rel, tails = prepare_batch()
    tails_len = tails.apply(len)
    
    idx = 0
    
    for i in tqdm.tqdm(range(tails_len.shape[0]), total=tails_len.shape[0]):
        next_idx = idx + tails_len[i]
        mini_pred = scores[idx:next_idx]
        sort_res = torch.argsort(-mini_pred)
        tails[i] = [tails[i][x] for x in sort_res]
        idx = next_idx

    tails.to_json(f"test.json")
    