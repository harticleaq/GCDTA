
def train(model, predictor, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, lr, epoch,
          batch_size, drug_pos, target_pos):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    predictor.train()
    LOG_INTERVAL = 10
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, chain(model.parameters(), predictor.parameters())), lr=lr, weight_decay=0)
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        ssl_loss, drug_embedding, target_embedding = model(drug_graph_batchs,
                                                                  target_graph_batchs, drug_pos, target_pos)
        output, _ = predictor(data.to(device), drug_embedding, target_embedding)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device)) + ssl_loss
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))


def test(model, predictor, device, loader, drug_graphs_DataLoader, target_graphs_DataLoader,drug_pos,
         target_pos):
    model.eval()
    predictor.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))  # drug graphs
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))  # target graphs
    with torch.no_grad():
        for data in loader:
            _, drug_embedding, target_embedding = model(drug_graph_batchs, target_graph_batchs, drug_pos, target_pos)
            output, _ = predictor(data.to(device), drug_embedding, target_embedding)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def train_predict(new_args):
    print("Data preparation in progress for the {} dataset...".format(args.dataset))
    num_drug, num_target = 2111, 228
    affinity_mat = load_data(args.dataset)
    train_data, test_data, drug_pos, target_pos = process_data(affinity_mat, args.dataset, args.num_pos, args.pos_threshold)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    drug_graphs_dict = get_drug_molecule_graph(
        json.load(open(f'data/{args.dataset}/drugs.txt'), object_pairs_hook=OrderedDict))
    drug_graphs_Data = GraphDataset(graphs_dict=drug_graphs_dict, dttype="drug")
    drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=collate,
                                                         batch_size=num_drug)
    target_graphs_dict = get_target_molecule_graph(
        json.load(open(f'data/{args.dataset}/targets.txt'), object_pairs_hook=OrderedDict), args.dataset)
    target_graphs_Data = GraphDataset(graphs_dict=target_graphs_dict, dttype="target")
    target_graphs_DataLoader = torch.utils.data.DataLoader(target_graphs_Data, shuffle=False, collate_fn=collate,
                                                           batch_size=num_target)

    print("Model preparation... ")
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    model = CSCoDTA(tau=args.tau,
                    lam=args.lam,
                    ns_dims=[num_drug + num_target + 2, 512, 256],
                    d_ms_dims=[78, 78, 78 * 2, 256],
                    t_ms_dims=[54, 54, 54 * 2, 256],
                    embedding_dim=128,
                    dropout_rate=args.edge_dropout_rate,
                    args=new_args)
    predictor = PredictModule()
    drug_pos = drug_pos.to(device)
    target_pos = target_pos.to(device)
    model.to(device)
    predictor.to(device)

    print("Start training...")
    for epoch in range(args.epochs):
        train(model, predictor, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, args.lr, epoch+1,
              args.batch_size, drug_pos, target_pos)
        G, P = test(model, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader,
                    drug_pos, target_pos)

        r = model_evaluate(G, P)
        # print(r)

    print('\npredicting for test data')
    G, P = test(model, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader, 
                drug_pos, target_pos)
    result = model_evaluate(G, P)
    print("result:", result)


if __name__ == '__main__':
    import os
    import argparse
    import torch
    import json
    import warnings
    from collections import OrderedDict
    from torch import nn
    from itertools import chain
    from data_process import load_data, process_data, get_drug_molecule_graph, get_target_molecule_graph
    from util import GraphDataset, collate, model_evaluate
    from models import CSCoDTA, PredictModule
    from utils.config_tools import get_defaults_yaml_args, update_args
    
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='kiba')
    parser.add_argument('--epochs', type=int, default=2500)    # --kiba 3000
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--edge_dropout_rate', type=float, default=0.2)   # --kiba 0.
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--num_pos', type=int, default=3)    # --kiba 10
    parser.add_argument('--pos_threshold', type=float, default=8.0)
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="default",
        help="Algorithm name.",
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
    new_args = vars(args)  # convert to dict
    if new_args["load_config"] != "":  # load config from existing config file
        with open(new_args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        new_args["algo"] = all_config["main_args"]["algo"]
        new_args["exp_name"] = all_config["main_args"]["exp_name"]
        algo_args = all_config["algo_args"]
    else:  # load config from corresponding yaml file
        algo_args = get_defaults_yaml_args(new_args["algo"])
    new_args.update(algo_args)
    update_args(unparsed_dict, new_args)  # update args from command line
    # args = parser.parse_args()

    train_predict(new_args)




