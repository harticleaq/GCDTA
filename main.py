# -*- coding: utf-8 -*-
"""
@Author: Anqi Huang
@Time: 2024/5/19
"""

import torch
import numpy as np
import argparse
import json
# from sklearn import metrics
import os
from utils.config_tools import get_defaults_yaml_args, update_args, save_configs
from utils.predata import MyDataset, _n2t, SpecialDataset
from torch.utils.data import DataLoader
from models.model import New
from tqdm.auto import tqdm
from datetime import datetime
from utils import metrics
import time

torch.autograd.set_detect_anomaly(True)

# Evaluate the trained model.
def evaluate(model, data_loader, loss_func, log_file):
    model.eval()
    test_loss = 0
    outputs, targets = [], []
    with torch.no_grad():
        for _, (smi, seq, y, adj) in tqdm(enumerate(data_loader)):
            smi = _n2t(smi)
            seq = _n2t(seq)
            y = _n2t(y)
            adj = _n2t(adj)
            output = model(smi, seq, adj)
            loss = loss_func(output.reshape(-1), y.reshape(-1))
            test_loss += loss.item()
            outputs.append(output.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)

    test_loss /= len(data_loader.dataset)
    
    results = {
        'loss': test_loss,
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'R': metrics.CORR(targets, outputs),
    }

    log_file.write(
                    ", ".join(
                        [f"{k}:{v}" for k, v in results.items()]
                        )
                        + "\n"
                    )
    return results

def ablation(model, data_loaders, type):
    data_loader = data_loaders[type]
    model.eval()
    outputs = []
    ys = []
    with torch.no_grad():
        for _, (smi, seq, y, adj) in tqdm(enumerate(data_loader)):
            smi = _n2t(smi)
            seq = _n2t(seq)
            y = _n2t(y)
            ys.append(y)
            adj = _n2t(adj)
            output = model(smi, seq, adj)
            outputs.append(output)


def special_case(model, drug, target, device="cuda:0"):
    from utils.predata import label_seq, label_smiles, process_smiles
    with torch.no_grad():
        smi = label_smiles(drug)
        seq = label_seq(target)
        adj = process_smiles(drug)
        smi = torch.tensor(smi).to(device).unsqueeze(0)
        seq = torch.tensor(seq).to(device).unsqueeze(0)
        adj = torch.tensor(adj).to(device)
        output = model(smi, seq, adj)
        print(adj)
        print(output)

def main():
    # Main function, initital args.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="default",
        help="Algorithm name.",
    )

    parser.add_argument(
        "--exp_name", type=str, default="bioAI", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="default",
        help="Must set to get bio data path: default, GPCR",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="./result",
        help="If set, save model result to progress.txt",
    )
    parser.add_argument(
        "--special_case",
        type=bool,
        default=False,
        help="If set, start special case",
    )
    
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["exp_name"] = all_config["main_args"]["exp_name"]
        algo_args = all_config["algo_args"]
    else:  # load config from corresponding yaml file
        algo_args = get_defaults_yaml_args(args["algo"])
    args.update(algo_args)
    update_args(unparsed_dict, args)  # update args from command line
    
    # Initial seed.
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    
    # Initial Log file and save current experiment configs.
    if args["model_dir"] is not None:
        cur_result_path = args["model_dir"]
    else:
        curtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cur_result_path = os.path.join(args["result_path"], curtime)
    if not os.path.exists(cur_result_path):
        os.makedirs(cur_result_path, exist_ok=True)
    log_file = open(
            os.path.join(cur_result_path, "progress.txt"), "w", encoding="utf-8"
        )
    save_configs(args, cur_result_path)
    
    # Initial data loader.
    if args["dataset"] == "default":
        args["data_path"] = "./data"
        data_loaders = {phase_name:
                    DataLoader(MyDataset(args["data_path"], phase_name,
                                         args["max_seq_len"], args["max_smi_len"], args["seq_size"]),
                               batch_size=args["batch_size"],
                               pin_memory=True,
                               num_workers=8,
                               shuffle=True)
                for phase_name in ['training', "validation"]}

    # Initital loss and model.
    args["loss_function"] = "mse"
    if args["loss_function"] == "mse":
        loss_func = torch.nn.MSELoss(reduction='sum')
    elif args["loss_function"] == "ce":
        loss_func = torch.nn.CrossEntropyLoss()
    model = New(args)
    model.to(args["device"])
    if args["model_dir"] is not None:
        state_dic = os.path.join(args["model_dir"], "best_model.pt") 
        model.load_state_dict(torch.load(state_dic))
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args["lr"], epochs=args["total_epochs"],
                                          steps_per_epoch=len(data_loaders['training'])) 
    best_loss = 1e10

    if args["special_case"]:
        # special_case(model, drug=drug, target=target)
        data_loaders = {phase_name:
                    DataLoader(MyDataset(args["data_path"], phase_name,
                                         args["max_seq_len"], args["max_smi_len"], args["seq_size"]),
                               batch_size=args["batch_size"],
                               pin_memory=True,
                               num_workers=8,
                               shuffle=True)
                for phase_name in ['test', "test71", "test105"]}
        ablation(model, data_loaders, "test105")

    else:
        # Start training.
        start_time = datetime.now()
        print("* Start training, -------> time: {:}".format(start_time))
        for epoch in range(args["total_epochs"]):
            tbar = tqdm(enumerate(data_loaders['training']), total=len(data_loaders['training']))
            for idx, (smi, seq, y, adj) in tbar:
                smi = _n2t(smi)
                seq = _n2t(seq)
                adj = _n2t(adj)
                y = _n2t(y)
                model.train()
                optimizer.zero_grad()
                output = model(smi, seq, adj)
                loss = loss_func(output.reshape(-1), y.reshape(-1))
                loss.backward() 
                optimizer.step()
                scheduler.step()
                tbar.set_description(f'  Train Epoch {epoch} Loss={loss.item() / len(y):.3f}')

            # Valid experiment.
            valid_results = evaluate(model, data_loaders["validation"], loss_func, log_file)
            print(f"  Valid Log: {valid_results}")
            if float(valid_results['RMSE']) < best_loss:
                torch.save(model.state_dict(), os.path.join(cur_result_path, 'best_model.pt'))
            
        # Test experiment.
        model.load_state_dict(torch.load(os.path.join(cur_result_path, 'best_model.pt')))
        if "validation" in data_loaders:
            train_results = evaluate(model, data_loaders["training"], loss_func, log_file)
            print(f"  Train Log: {train_results}")
            valid_results = evaluate(model, data_loaders["validation"], loss_func, log_file)
            print(f"  Valid Log: {valid_results}")
        # test_results = evaluate(model, data_loaders["test"], loss_func, log_file)
        # print(f"  Test Log: {test_results}")   
        if "test71" and "test105" in data_loaders:
            test71 = evaluate(model, data_loaders["test71"], loss_func, log_file)
            test105 = evaluate(model, data_loaders["test105"], loss_func, log_file)
            print(f"  Test71 Log: {test71}")
            print(f"  Test105 Log: {test105}")   
        end_time = datetime.now()
        print("* Finish training, -------> time: {:}".format(end_time))
                


if __name__ == "__main__":
    main()
