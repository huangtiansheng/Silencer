import torch
import utils
import models
import math
import copy
import numpy as np
from agent import Agent
from agent_sparse import Agent as Agent_s
from tqdm import tqdm
from options import args_parser
from aggregation import Aggregation
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
from time import ctime
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from utils import H5Dataset
import sys
from resnet9 import ResNet9_tinyimagenet, ResNet9
import random
from models import CNN_MNIST, get_model

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
import logging


def CLP(net, u):
    sparse_num = 0
    total = 0
    params = net.state_dict()
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            channel_lips = []
            weights_norm = []
            for idx in range(m.weight.shape[0]):
                weight = m.weight[idx]
                weight = weight.reshape(weight.shape[0], -1).cpu()
                channel_lips.append(torch.svd(weight)[1].max())
                weights_norm.append(float(torch.norm(weight)))
            channel_lips = torch.Tensor(channel_lips)

            index = torch.where(channel_lips > channel_lips.mean() + u * channel_lips.std())[0]
            params[name + '.weight'][index] = 0
            sparse_num += torch.numel(params[name + '.weight'][index])
            # print(index)
            total += torch.numel(m.weight)

    print(sparse_num / total)
    print(channel_lips)
    weights_norm = np.round(np.sort(weights_norm), decimals=4).tolist()
    print(weights_norm)
    net.load_state_dict(params)


def WMP(net, k):
    sparse_num = 0
    total = 0
    params = net.state_dict()
    for name in params:
        x = params[name].view(-1)
        # Sort the tensor in ascending order
        sorted_indices = torch.argsort(x)
        prune = int(k * torch.numel(x))
        # Get the indices of the lowest k elements
        pruned_indices = sorted_indices[:prune]
        mask = torch.ones_like(x)
        # Set the values at the pruned indices to 0
        mask[pruned_indices] = 0
        x.data *= mask

    # print(sparse_num/total)
    net.load_state_dict(params)


def GMP(net, gradient, k):
    sparse_num = 0
    total = 0
    params = net.state_dict()
    for name in gradient:
        # print(name)
        x = torch.abs(gradient[name].view(-1))
        # Sort the tensor in ascending order
        sorted_indices = torch.argsort(x, descending=True)
        prune = int(k * torch.numel(x))
        # Get the indices of the lowest k elements
        pruned_indices = sorted_indices[:prune]
        mask = torch.ones_like(x)
        # Set the values at the pruned indices to 0
        mask[pruned_indices] = 0
        flat_param = params[name].view(-1)
        flat_param *= mask

    # print(sparse_num/total)
    net.load_state_dict(params)


def GMDP(net, gradient, k):
    sparse_num = 0
    total = 0
    params = net.state_dict()
    for name in gradient:
        # print(name)
        x = torch.abs(gradient[name].view(-1))
        # Sort the tensor in ascending order
        sorted_indices = torch.argsort(x, descending=True)
        prune = int(k * torch.numel(x))
        # Get the indices of the lowest k elements
        pruned_indices = sorted_indices[:prune]
        mask = torch.ones_like(x)
        # Set the values at the pruned indices to 0
        mask[pruned_indices] = 0
        flat_param = params[name].view(-1)
        noise = torch.tensor(np.random.normal(0, 1, flat_param.size()), dtype=torch.float).to(flat_param.device)
        flat_param += (1-mask) *noise
        # flat_param *=  mask
    # print(sparse_num/total)
    net.load_state_dict(params)


def GMP_C(net, gradients, k):
    sparse_num = 0
    total = 0
    params = net.state_dict()
    for name in gradients[0]:
        for index, gradient in enumerate(gradients):
            x = torch.abs(gradient[name].view(-1))
            # Sort the tensor in ascending order
            sorted_indices = torch.argsort(x, descending=True)
            prune = int(k * torch.numel(x) / 40)
            # Get the indices of the lowest k elements
            pruned_indices = sorted_indices[:prune]
            mask = torch.ones_like(x)
            # Set the values at the pruned indices to 0
            mask[pruned_indices] = 0
            flat_param = params[name].view(-1)
            flat_param *= mask

    # print(sparse_num/total)
    net.load_state_dict(params)


def RP(net, k):
    params = net.state_dict()
    for name in params:
        # print(name)
        if "bn" not in name:
            prune = int(k * torch.numel(params[name]))
            mask = torch.ones_like(params[name].view(-1))
            if prune > 0:
                temp = torch.ones_like(params[name].view(-1))
                pruned_indices = torch.multinomial(temp, prune, replacement=False)
                # Set the values at the pruned indices to 0
                mask[pruned_indices] = 0

            flat_param = params[name].view(-1)
            flat_param *= mask
    # print(sparse_num/total)
    net.load_state_dict(params)


def warm_up_bn(net, data_loader):
    net.train()
    for _ in range(2):
        # start = time.time()
        for _, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device="cuda:0"), \
                             labels.to(device="cuda:0")
            outputs = net(inputs)


def prune_test(dict, val_loader, poisoned_val_loader, args):
    # percents = [1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]
    if args.prune_method == "GMP-C":
        if args.data == "tinyimagenet":
            percents = [0, 0.00001, 0.0001, 0.0005, 0.001, 0.0015, 0.002]
        else:
            percents = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]
    elif args.prune_method == "RP":
        if args.data == "tinyimagenet":
            percents = [0, 0.01, 0.025, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2]
        else:
            percents = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    elif args.prune_method == "GMP-M":
        if args.data == "tinyimagenet":
            percents = [0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.03, 0.05, 0.075]
        else:
            percents = [0, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    else:
        if args.data == "tinyimagenet":
            percents = [0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.03, 0.05, 0.075, 0.1]
        else:
            percents = [0, 0.0025, 0.005, 0.01, 0.015, 0.03, 0.05, 0.1, 0.15, 0.2]

    ASR = []
    accuracy = []
    # percents= [0.01,0.02,0.03 ]
    global_model = get_model(args.data).to(args.device)
    vec = dict['clean_model']
    state_dict = utils.vector_to_model(vec, global_model)
    global_model.load_state_dict(state_dict)
    # warm_up_bn(global_model, train_loader)
    if args.prune_method == "GMP-M":
        gradient = agents[22].screen_gradients(global_model)
    elif args.prune_method == "GMP-B":
        gradient = agents[23].screen_gradients(global_model)
    elif args.prune_method == "GMP-C":
        gradient = []
        for agent in agents:
            gradient += [agent.screen_gradients(global_model)]

    for percent_to_select in percents:
        # logging.info("percentage  {}".format(percent_to_select))

        # CLP(global_model, percent_to_select)
        # WMP(global_model, percent_to_select)
        if args.prune_method == "RP":
            trail = 5
        else:
            trail = 1
        temp1 = []
        temp2 = []
        for random_exp in range(trail):
            global_model.load_state_dict(state_dict)
            if args.prune_method == "GMP-M" or args.prune_method == "GMP-B":
                GMP(global_model, gradient, percent_to_select)
            elif args.prune_method == "GMP-C":
                GMP_C(global_model, gradient, percent_to_select)
            else:
                RP(global_model, percent_to_select)
            # warm_up_bn(global_model, train_loader)
            criterion = nn.CrossEntropyLoss().to(args.device)

            val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(global_model, criterion, val_loader,
                                                                                  args, 0, num_classes=num_target)
            temp1 += [val_acc]
            # logging.info(f'| val_acc: {val_loss:.3f} / {val_acc:.3f} |')

            poison_loss, (poison_acc, _), _ = utils.get_loss_n_accuracy(global_model, criterion,
                                                                        poisoned_val_loader,
                                                                        args, 0, num_classes=num_target)
            temp2 += [poison_acc]
            # logging.info(f'|  Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')
            # ask the guys to finetune the classifier
            # logging.info(mask_aggrement)
            # logging.info(temp2)
        accuracy += [round(np.mean(temp1), 2)]
        ASR += [round(np.mean(temp2), 2)]
    logging.info(ASR)
    logging.info(accuracy)

def dp_test(dict, val_loader, poisoned_val_loader, args):
    # percents = [1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]
    if args.prune_method == "GMP-C":
        if args.data == "tinyimagenet":
            percents = [0, 0.00001, 0.0001, 0.0005, 0.001, 0.0015, 0.002]
        else:
            percents = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]
    elif args.prune_method == "RP":
        if args.data == "tinyimagenet":
            percents = [0, 0.01, 0.025, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2]
        else:
            percents = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    elif args.prune_method == "GMP-M":
        if args.data == "tinyimagenet":
            percents = [0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.03, 0.05, 0.075]
        else:
            percents = [0, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    else:
        if args.data == "tinyimagenet":
            percents = [0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.03, 0.05, 0.075, 0.1]
        else:
            percents = [0, 0.0025, 0.005, 0.01, 0.015, 0.03, 0.05, 0.1, 0.15, 0.2]

    ASR = []
    accuracy = []
    # percents= [0.01,0.02,0.03 ]
    global_model = get_model(args.data).to(args.device)
    vec = dict['clean_model']
    state_dict = utils.vector_to_model(vec, global_model)
    global_model.load_state_dict(state_dict)
    # warm_up_bn(global_model, train_loader)
    if args.prune_method == "GMP-M":
        gradient = agents[22].screen_gradients(global_model)
    elif args.prune_method == "GMP-B":
        gradient = agents[23].screen_gradients(global_model)
    elif args.prune_method == "GMP-C":
        gradient = []
        for agent in agents:
            gradient += [agent.screen_gradients(global_model)]

    for percent_to_select in percents:
        # logging.info("percentage  {}".format(percent_to_select))

        # CLP(global_model, percent_to_select)
        # WMP(global_model, percent_to_select)
        if args.prune_method == "RP":
            trail = 5
        else:
            trail = 1
        temp1 = []
        temp2 = []
        for random_exp in range(trail):
            global_model.load_state_dict(state_dict)
            if args.prune_method == "GMP-M" or args.prune_method == "GMP-B":
                GMDP(global_model, gradient, percent_to_select)
            elif args.prune_method == "GMP-C":
                GMP_C(global_model, gradient, percent_to_select)
            else:
                RP(global_model, percent_to_select)
            # warm_up_bn(global_model, train_loader)
            criterion = nn.CrossEntropyLoss().to(args.device)

            val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(global_model, criterion, val_loader,
                                                                                  args, 0, num_classes=num_target)
            temp1 += [val_acc]
            # logging.info(f'| val_acc: {val_loss:.3f} / {val_acc:.3f} |')

            poison_loss, (poison_acc, _), _ = utils.get_loss_n_accuracy(global_model, criterion,
                                                                        poisoned_val_loader,
                                                                        args, 0, num_classes=num_target)
            temp2 += [poison_acc]
            # logging.info(f'|  Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')
            # ask the guys to finetune the classifier
            # logging.info(mask_aggrement)
            # logging.info(temp2)
        accuracy += [round(np.mean(temp1), 2)]
        ASR += [round(np.mean(temp2), 2)]
    logging.info(ASR)
    logging.info(accuracy)


def weights_visualization(dict):
    global_model = CNN_MNIST().to(args.device)
    # gradients = []
    # for _id in range(0, args.num_agents):
    #     gradients += [agents[_id].screen_gradients(global_model)['conv1.weight']]
    # plot_polar(gradients)

    vec = dict['clean_model']
    state_dict = utils.vector_to_model(vec, global_model)
    global_model.load_state_dict(state_dict)
    gradients = []
    for _id in range(0, args.num_agents):
        gradients += [ agents[_id].screen_gradients(global_model)['conv1.weight']**2*state_dict['conv1.weight']**2]
    plot_polar(gradients)

def weights_visualization2():
    chosen_clients = [22, 20, 30, 31]
    chosen_rounds = [ 100, 200, 300 ]

    global_model = CNN_MNIST().to(args.device)
    # gradients = []
    # for _id in range(0, args.num_agents):
    #     gradients += [agents[_id].screen_gradients(global_model)['conv1.weight']]
    # plot_polar(gradients)
    gradients = [[] for index, i in enumerate(chosen_rounds) ]
    for index, i in enumerate(chosen_rounds):
        PATH = "checkpoint/Ack4_40_fedavg_fmnist_alpha0.5_Epoch2_inject0.5_dense0.25_Aggavg_sm1_noniidFalse_theta20_attackbadnet_end1000_af0.0001_ns8_toprandom_rnd{}".format(i)
        dict = torch.load(PATH)
        vec = dict['clean_model']
        state_dict = utils.vector_to_model(vec, global_model)
        global_model.load_state_dict(state_dict)
        for _id in chosen_clients:
            gradients[index] += [agents[_id].screen_gradients(global_model)['conv1.weight']]
    plot_polar2(gradients, chosen_rounds, chosen_clients)

def diagonal_hessian_visualization(dict):
    global_model = CNN_MNIST().to(args.device)
    vec = dict['clean_model']
    state_dict = utils.vector_to_model(vec, global_model)
    global_model.load_state_dict(state_dict)
    # gradients = agents[0].screen_gradients(global_model)
    # vector = parameters_to_vector(
    #     [gradients[name] for name in gradients]).detach()
    digonal_hessian = agents[0].screen_hessian(global_model)

def plot_polar2(gradient, chosen_round, chosen_client):
    import numpy as np
    import matplotlib.pyplot as plt
    colors = [[171 / 255, 29 / 255, 34 / 255], [219 / 255, 155 / 255, 52 / 255], [168 / 255, 132 / 255, 98 / 255],
              [16 / 255, 139 / 255, 150 / 255]]
    # Update magnitudes for parameters of the channel neuron (example values)
    alphas = [1, 0.8, 0.8, 0.7, 0.6, 0.5]

    fig = plt.figure(figsize=(4, 6))

    for index2, j in enumerate(chosen_client):
        for index1, i in enumerate(chosen_round):
            flat_gradient = torch.abs(gradient[index1][index2][2].view(-1))
            ax = fig.add_subplot(len(chosen_client),len(chosen_round), index2 * len(chosen_round) + index1 + 1, projection='polar')
            angles = np.linspace(0, 2 * np.pi, len(flat_gradient), endpoint=False)
            bar_width = 0.2
            ax.bar(angles, flat_gradient.cpu(), width=bar_width, color=colors[index2], alpha=alphas[index1])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(linestyle='--', linewidth=1)
            # Set plot limits and title
            # ax.set_ylim(0, 0.5)  # Adjust the y-axis limit to match your magnitude range
    plt.tight_layout()
    plt.savefig("fig/grad2.png", dpi=500)
    plt.show()

def plot_polar(gradient):
    import numpy as np
    import matplotlib.pyplot as plt
    colors = [[171 / 255, 29 / 255, 34 / 255], [219 / 255, 155 / 255, 52 / 255], [168 / 255, 132 / 255, 98 / 255],
              [16 / 255, 139 / 255, 150 / 255]]
    # Update magnitudes for parameters of the channel neuron (example values)
    alphas = [1, 0.8, 0.8, 0.7, 0.6, 0.5]

    chosen_neuron = [2, 5, 10]
    chosen_client = [22, 20, 30, 31]
    fig = plt.figure(figsize=(4, 6))
    for index2, j in enumerate(chosen_client):
        for index1, i in enumerate(chosen_neuron):
            flat_gradient = torch.abs(gradient[j][i].view(-1))
            # Create a figure with polar projectio
            ax = fig.add_subplot( len(chosen_client),len(chosen_neuron), index2 * len(chosen_neuron) + index1 + 1, projection='polar')
            # Arrange update magnitudes radially
            angles = np.linspace(0, 2 * np.pi, len(flat_gradient), endpoint=False)
            # Define the bar width
            bar_width = 0.2
            # Plot the fan-out bars
            ax.bar(angles, flat_gradient.cpu(), width=bar_width, color=colors[index2], alpha=alphas[index1])
            # Remoe the x-tick labels
            ax.set_xticklabels([])
            # Remove the degree labels from radial ticks
            ax.set_yticklabels([])
            ax.grid( linestyle = '--', linewidth = 1)
            # Set plot limits and title
            # ax.set_ylim(0, 0.5)  # Adjust the y-axis limit to match your magnitude range
    plt.tight_layout()
    plt.savefig("fig/grad1.png", dpi=500)
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
    PATH = "checkpoint/Ack4_40_fedavg_fmnist_alpha0.5_Epoch2_inject0.5_dense0.25_Aggavg_sm1_noniidFalse_theta20_attackbadnet_end1000_af0.0001_ns8_toprandom"
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

    # args.prune_method = "GMP-C"
    # logging.info(args.prune_method)
    # prune_test(dict, val_loader, poisoned_val_loader, args)
    #
    args.prune_method = "GMP-M"
    logging.info(args.prune_method)
    # prune_test(dict, val_loader, poisoned_val_loader, args)
    dp_test(dict, val_loader, poisoned_val_loader, args)
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


