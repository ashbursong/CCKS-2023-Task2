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
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    
    working_dir = os.path.join(os.path.expanduser(cfg.output_dir), cfg.model["class"], cfg.dataset["class"], cfg.exp_name)
    device = util.get_device(cfg)
    
    dataset = util.build_dataset(cfg)
    _, valid_data = dataset[0], dataset[1]
    valid_data = valid_data.to(device)
    filtered_data = None
    
    cfg.model.num_relation = dataset.num_relations
    model = util.build_model(cfg)
    state = torch.load(f"{working_dir}/best_model.pth", map_location=device)
    model.load_state_dict(state['model'])

    batch, head_rel, tails = prepare_batch()
    tails_len = tails.apply(len)

    batch = batch.to(device)
    model = model.to(device)
    
    idx = 0
    pred = []
    
    for i in tqdm.tqdm(range(tails_len.shape[0]), total=tails_len.shape[0]):
        next_idx = idx + tails_len[i]
        mini_batch = batch[idx:next_idx].unsqueeze(0)
        mini_pred = model(valid_data, mini_batch).squeeze(0).cpu().detach()
        pred.append(mini_pred)
        sort_res = torch.argsort(-mini_pred)
        tails[i] = [tails[i][x] for x in sort_res]
        idx = next_idx

    tails.to_json(f"{working_dir}/test.json")
    pred = torch.cat(pred)
    torch.save(pred, f"{working_dir}/scores.pt")
    