import torch
import utils
import copy
import numpy as np
from agent import Agent
from options import args_parser
from torch.utils.data import DataLoader
import torch.nn as nn
import random
from models import CNN_MNIST, get_model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
import logging




def load_model(dict, args):
    global_model = get_model(args.data).to(args.device)
    vec = dict['clean_model']
    if args.method == "fedbi":
        masks = dict["masks"]
        state_dict = utils.vector_to_model(vec, global_model)
        consensus = {}
        for name, param in global_model.named_parameters():
            mask = 0
            for id in enumerate(args.num_agents):
                mask += masks[name].to(args.device)
            consensus[name] = torch.where(mask.to(args.device) >= args.theta, torch.ones_like(param),
                                          torch.zeros_like(param))
            state_dict[name] *= consensus[name]
        global_model.load_state_dict(state_dict)
    else:
        state_dict = utils.vector_to_model(vec, global_model)
        global_model.load_state_dict(state_dict)
    return global_model

def visual_test_tsne(dict, val_dataset, poisoned_dataset, args):
    # percents = [1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]
    global_model = load_model(dict, args)
    # remove linear layer
    global_model = torch.nn.Sequential(*list(global_model.children())[:-2])

    def extract_features(dataset):
        features = []
        for images, _ in dataset:
            with torch.no_grad():
                features.append(global_model(images).squeeze().numpy())
        return features


    # Extract features
    num_poisoned=100
    random_indices1 = random.sample(range(0, len(val_dataset)), 1000)
    random_indices2 = random.sample(range(0, len(poisoned_dataset)), num_poisoned)
    features1= extract_features(val_dataset[random_indices1][0])
    features2 = extract_features(poisoned_dataset[random_indices2][0])
    
    
    # Merge the features by concatenating them
    merged_features = np.concatenate([features1, features2], axis=0)
    merged_labels  = np.concatenate([val_dataset[random_indices1][1], [ 10 for i in range(num_poisoned)]], axis=0)
    # Reduce dimensions using t-SNE
    tsne = TSNE(n_components=10, random_state=42)
    embedded_features = tsne.fit_transform(merged_features)

    # Plot the t-SNE visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(embedded_features[:, 0], embedded_features[:, 1], c=merged_labels, cmap='viridis')
    plt.title('t-SNE Visualization of ResNet-9 Features')
    plt.colorbar()
    plt.show()



if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    args = args_parser()
    args.num_agents = 40
    num_target = 200
    args.data = "fmnist"
    args.non_iid = False

    args.poison_frac = 0.5
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    logPath = "logs"
    logging.info(args)
    # PATH= "checkpoint/AckRatio4_1_Methodfedavg_datacifar10_alpha0.1_Rnd200_Epoch2_inject0.5_dense0.5_Aggavg_se_threshold0.0001_noniidFalse_maskthreshold20_attackr_neurotoxin.pt"
    PATH = "checkpoint/Ack4_40_fedavg_fmnist_alpha0.5_Epoch2_inject0.5_dense0.25_Aggavg_sm1_noniidFalse_theta20_attackbadnet_end1000_af0.0001_ns8_toprandom_rnd300.pt"
    dict = torch.load(PATH)

    # load dataset and user groups (i.e., user to data mapping)
    train_dataset, val_dataset = utils.get_datasets(args.data)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                            pin_memory=False)
    # fedemnist is handled differently as it doesn't come with pytorch
    if args.data != 'fedemnist':
        if args.non_iid:
            user_groups = utils.distribute_data_dirichlet(train_dataset, args)
        else:
            user_groups = utils.distribute_data(train_dataset, args, n_classes=num_target)

    # train_loader = DataLoader(poisoned_val_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
    #                                  pin_memory=False)
    # poison the validation dataset
    idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
    if args.data != "tinyimagenet":
        # poison the validation dataset
        poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
    else:
        poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=args)
    poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                                     pin_memory=False)

    # poison the validation dataset
    # logging.info(idxs)
    if args.data != "tinyimagenet":
        idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
        poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        utils.poison_dataset(poisoned_val_set_only_x.dataset, args, idxs, poison_all=True, modify_label=False)
    else:
        poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=args,
                                                     modify_label=False)

    poisoned_val_only_x_loader = DataLoader(poisoned_val_set_only_x, batch_size=args.bs, shuffle=False,
                                            num_workers=args.num_workers,
                                            pin_memory=False)

    agents, agent_data_sizes = [], {}
    if args.data == "tinyimagenet":
        corrupt_idx = [0, 1, 2, 3]
    else:
        corrupt_idx = [22 ,20 ,25  ,4]
    for _id in range(0, args.num_agents):
        agents += [Agent(_id, args, train_dataset, user_groups[_id], corrupt_idx=corrupt_idx)]



    visual_test_tsne(dict, val_dataset, poisoned_val_set, args)


    # args.prune_method = "GMP-C"
    # logging.info(args.prune_method)
    # prune_test(dict, val_loader, poisoned_val_loader, args)
    #

    # args.prune_method = "GMP-M"
    # logging.info(args.prune_method)
    # # prune_test(dict, val_loader, poisoned_val_loader, args)
    # dp_test(dict, val_loader, poisoned_val_loader, args)

    #
    # args.prune_method = "GMP-B"
    # logging.info(args.prune_method)
    # prune_test(dict, val_loader, poisoned_val_loader, args)
    #
    # args.prune_method = "RP"
    # logging.info(args.prune_method)
    # prune_test(dict, val_loader, poisoned_val_loader, args)

    # weights_visualization(dict)
    # diagonal_hessian_visualization(dict)
    # weights_visualization2()


