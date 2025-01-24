import utils
import models
import math
import copy
import numpy as np
from agent import Agent
from agent_bi import Agent as Agent_bi
from options import args_parser
from aggregation import Aggregation
import torch
import random
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
import logging


def _benefit_choose( cur_clnt, client_num_in_total, client_num_per_round, topology="random"):
    if client_num_in_total == client_num_per_round:
        # If one can communicate with all others and there is no bandwidth limit
        client_indexes = [client_index for client_index in range(client_num_in_total)]
        return client_indexes

    if topology == "random":
        # Random selection of available clients
        num_clients = min(client_num_per_round, client_num_in_total)
        client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        while cur_clnt in client_indexes:
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)

    elif topology == "ring":
        # Ring Topology in Decentralized setting
        left = (cur_clnt - 1 + client_num_in_total) % client_num_in_total
        right = (cur_clnt + 1) % client_num_in_total
        client_indexes = np.asarray([left, right])

    elif topology == "full":
        # Fully-connected Topology in Decentralized setting
        client_indexes = np.array([ i for i in range(client_num_in_total)]).squeeze()
        client_indexes = np.delete(client_indexes, int(np.where(client_indexes == cur_clnt)[0]))
    return client_indexes

def warm_up_bn(net, data_loader):
    net.train()
    for _ in range(2):
        # start = time.time()
        for _, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device="cuda:0"), \
                                labels.to(device="cuda:0")
            outputs = net(inputs)
    

def construct_personalized_val(val_dataset, agents, n_class=10):
    # hts
    
    test_y = copy.deepcopy(val_dataset.targets)
    test_y = test_y.numpy()
    # print(test_y.shape)
    test_idx_list = [np.where(test_y == i)[0] for i in range(n_class)]
    test_num = 100
    client_test_ids = [np.zeros((test_num)).astype(np.int64) for client__ in
                        range(len(agents))]
    already_alloc = [0 for i in range(len(agents))]
    client_test_loader_list = []
    client_per_backdoor_loader_list= []
    for client__ in range(len(agents)):
        client_y = agents[client__].train_dataset.targets
        print(client_y)
        client_y = client_y.numpy()
        # print(client_y.shape)
        # client_class_num = [ np.sum(self.client_y[client__]==cls) for cls in range(self.n_cls)]
        for label in range(n_class):
            porportion = np.mean(client_y==label )
            # print(porportion)

            label_num = math.floor(porportion * test_num)
            rand_perm = np.random.permutation(len(test_idx_list[label]))
            # print(client_test_ids[0].shape)
            # print(test_idx_list[label][rand_perm[:label_num]].shape)
            client_test_ids[client__][already_alloc[client__]:already_alloc[client__] + label_num] = test_idx_list[label][rand_perm[:label_num]]
            already_alloc[client__] += label_num
        client_test_ids[client__] = np.delete(client_test_ids[client__], slice(already_alloc[client__], test_num), axis=0)
        
        
        
        
        client_test_dataset = utils.DatasetSplit(copy.deepcopy(val_dataset), client_test_ids[client__] )
        print("Client %3d: " %client__ + 
                ', '.join(["%.3f" %np.mean(client_y==cls) for cls in range(10)]) + 
                ', Amount:%d' %client_y.shape)
        print(client_test_dataset.targets)
        
        
        
        client_test_loader_list += [ DataLoader(client_test_dataset, batch_size=args.bs, shuffle=False,
                                            num_workers=args.num_workers,
                                            pin_memory=False) ]
        idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
        all_idxs = list(set(client_test_ids[client__]).intersection(idxs))
        client_backdoor_dataset = utils.DatasetSplit(copy.deepcopy(val_dataset), all_idxs)
        utils.poison_dataset(client_backdoor_dataset.dataset, args,  all_idxs, poison_all=True)
        client_per_backdoor_loader_list += [ DataLoader(client_backdoor_dataset, batch_size=args.bs, shuffle=False,
                                            num_workers=args.num_workers,
                                            pin_memory=False) ]

    #     count += client_y[client].shape[0]
    #     print("Client test %3d: " %client +
    #             ', '.join(["%.3f" %np.mean(self.client_test_y[client]==cls) for cls in range(s10)]) +
    #             ', Amount:%d' %client_test_y[client].shape[0])
    return client_test_loader_list, client_per_backdoor_loader_list


def do_test(rnd, global_model, rnd_global_params, args, corrupt_idx, clients=None,identity_grid=None, noise_grid=None):
     # inference in every args.snap rounds
        if rnd==1 or rnd % args.snap == 0:
            chosen = np.random.choice(args.num_agents, math.floor(args.num_agents), replace=False)
            test_model = copy.deepcopy(global_model)
            test_model.to(args.device)
            acc_vec = [] 
            asr_vec = []
            bacc_vec= []
            fp_vec = []
            gradient_norms = []
            for client in range(args.num_agents):
                if client not in corrupt_idx or len(corrupt_idx)==args.num_agents:
                    state_dict = utils.vector_to_model(copy.deepcopy(rnd_global_params[client]), test_model)
                    test_model.load_state_dict(state_dict)
                    val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(test_model, criterion, val_loader,
                                                                                        args, rnd, num_target)
                    acc_vec.append(val_acc)

                    poison_loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(test_model, criterion,
                                                                                    poisoned_val_loader, args, rnd, num_target,identity_grid=identity_grid, noise_grid=noise_grid)
                    asr_vec.append(asr)
                    
                    _, (bacc, _), fail_samples = utils.get_loss_n_accuracy(test_model, criterion,
                                                                                        poisoned_val_only_x_loader, args, rnd,
                                                                                        num_target,identity_grid=identity_grid, noise_grid=noise_grid)
                    bacc_vec.append(bacc)
                    
                    _, (fp, _), fail_samples = utils.get_loss_n_accuracy(test_model, criterion,
                                                                                        fp_loader, args, rnd,
                                                                                        num_target,identity_grid=identity_grid, noise_grid=noise_grid)
                    fp_vec.append(fp)
                    
                    if args.debug:
                        break
                # logging.info(torch.norm(rnd_global_params[client]))
                # poison_loss, (poison_acc, _), fail_samples = utils.get_loss_n_accuracy(global_model, criterion,
                #                                                                        poisoned_val_only_x_loader, args,
                #                                                                        rnd, num_target)
                # pacc_vec.append(poison_acc)
                # logging.info(f'| Poison Loss/Poison accuracy: {poison_loss:.3f} / {poison_acc:.3f} |')
                # if args.method != "silencer":
                #     # get full train dataset's dataloader
                #     global_gradient =  {name: 0 for name, param in test_model.named_parameters()}
                #     for _, (x, labels) in enumerate(clean_loader):
                #         test_model.zero_grad()
                #         x, labels = x.to(args.device), labels.to(args.device)
                #         log_probs = test_model.forward(x)
                #         minibatch_loss = criterion(log_probs, labels.long())
                #         loss = minibatch_loss
                #         loss.backward()
                #         # when applying cross entropy, digonal of empirical Fisher information is the square of gradient
                #         for name, param in test_model.named_parameters():
                #             global_gradient [name] += param.grad.data
                            
                #     full_graident_norm = 0
                #     for name in global_gradient:
                #         full_graident_norm+= torch.norm(global_gradient[name]/len(clean_dataset))
                #     gradient_norms += [full_graident_norm.cpu()]
                
                
                
            # writer.add_scalar('Validation/Loss', val_loss, rnd)
            # writer.add_scalar('Validation/Accuracy', val_acc, rnd)
            logging.info("val: {}".format(acc_vec))
            logging.info("asr: {}".format(asr_vec))
            logging.info(f'| Val_Loss/Val_Acc: xx / {np.mean(acc_vec)*100:.2f} |')
            logging.info(f'| Attack Loss/Attack Success Ratio: xx / {np.mean(asr_vec)*100:.2f} |')
            logging.info(f'| Backdoor Acc:  {np.mean(bacc)*100:.2f} |')
            logging.info(f'| False Positive:  {np.mean(fp_vec)*100:.2f} |')
            # if args.method != "silencer":
            #     logging.info("gradient norm: {}".format(np.mean(gradient_norms)))
            
            

            if  args.method == "silencer":
                acc_vec = [] 
                asr_vec = []
                bacc_vec= []
                fp_vec = []
                gradient_norms = []
                
                consensus = {}
                import time 
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                # Record the start event
                start_event.record()
                for name, param in test_model.named_parameters():
                    mask = 0
                    for id, agent in enumerate(agents):
                        mask += agent.mask[name].to(args.device)
                    consensus[name] = torch.where(mask.to(args.device) >= args.theta, torch.ones_like(param),
                                                  torch.zeros_like(param))
                    logging.info(torch.sum(consensus[name]) / torch.numel(consensus[name]))
                end_event.record()
                # Wait for all CUDA operations to finish
                torch.cuda.synchronize()
                print("consensus fusion time {}".format(start_event.elapsed_time(end_event)/ 1000))
                for client in range(args.num_agents):
                    if client not in corrupt_idx or len(corrupt_idx)==args.num_agents:
                        state_dict = utils.vector_to_model(copy.deepcopy(rnd_global_params[client]), test_model)
                        test_model.load_state_dict(state_dict)
                        for name, param in test_model.named_parameters():
                            param.data *= consensus[name]
                        if args.data =="tinyimagenet":
                            warm_up_bn(test_model,clients[client].train_loader)
                        val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(test_model, criterion,
                                                                                            val_loader,
                                                                                            args, rnd, num_target)
                        acc_vec.append(val_acc)
                        poison_loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(test_model, criterion,
                                                                                        poisoned_val_loader, args, rnd,
                                                                                        num_target)
                        asr_vec.append(asr)
                        _, (bacc, _), fail_samples = utils.get_loss_n_accuracy(test_model, criterion,
                                                                                        poisoned_val_only_x_loader, args, rnd,
                                                                                        num_target)
                        bacc_vec.append(bacc)
                        
                        
                        _, (fp, _), fail_samples = utils.get_loss_n_accuracy(test_model, criterion,
                                                                                        fp_loader, args, rnd,
                                                                                        num_target)
                        fp_vec.append(fp)
                        
                        
    
                        if args.debug:
                            break
                logging.info("clean val: {}".format(acc_vec))
                logging.info("clean asr: {}".format(asr_vec))
                logging.info(f'| Clean Val_Loss/Val_Acc: xx / {np.mean(acc_vec)*100:.2f} |')
                logging.info(f'| Clean Attack Loss/Attack Success Ratio: xx / {np.mean(asr_vec)*100:.2f} |')
                logging.info(f'| Clean Backdoor Acc:  {np.mean(bacc_vec)*100:.2f} |')
                logging.info(f'| Clean False Positive:  {np.mean(fp_vec)*100:.2f} |')
                if args.personalized:
                    acc_vec = [] 
                    asr_vec = []
                    bacc_vec= []
                    fp_vec = []
                    gradient_norms = []
                    consensus = {}
                    for name, param in test_model.named_parameters():
                        mask = 0
                        for id, agent in enumerate(agents):
                            mask += agent.mask[name].to(args.device)
                        consensus[name] = torch.where(mask.to(args.device) >= args.theta, torch.ones_like(param),
                                                    torch.zeros_like(param))
                  
                    for client in range(args.num_agents):
                        if client not in corrupt_idx or len(corrupt_idx)==args.num_agents:
                            state_dict = utils.vector_to_model(copy.deepcopy(rnd_global_params[client]), test_model)
                            test_model.load_state_dict(state_dict)
                            for name, param in test_model.named_parameters():
                                param.data *= agents[client].mask[name].to(args.device)
                            if args.data =="tinyimagenet":
                                warm_up_bn(test_model,clients[client].train_loader)
                            val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(test_model, criterion,
                                                                                                client_per_test_loader_list[client],
                                                                                                args, rnd, num_target)
                            acc_vec.append(val_acc)
                            poison_loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(test_model, criterion,
                                                                                            client_per_backdoor_loader_list[client], args, rnd,
                                                                                            num_target)
                            asr_vec.append(asr)
                    logging.info(f'| Clean personalized Val_Loss/Val_Acc: xx / {np.mean(acc_vec)*100:.2f} |')
                    logging.info(f'| Clean personalized Attack Loss/Attack Success Ratio: xx / {np.mean(asr_vec)*100:.2f} |')
                # logging.info("gradient norm: {}".format(np.mean(gradient_norms)))
          
            
def save_checkpoints(rnd, args, agents, rnd_global_params, corrupt_idx): 
    save_frequency = 100
    if rnd % save_frequency == 0:
        if args.mask_init == "uniform":
            PATH = "checkpoint/uniform_AckRatio{}_{}_Method{}_data{}_alpha{}_Rnd{}_Epoch{}_inject{}_dense{}_Agg{}_se_threshold{}_noniid{}_maskthreshold{}_attack{}.pt".format(
                args.num_corrupt, args.num_agents, args.method, args.data, args.alpha, rnd, args.local_ep,
                args.poison_frac, args.dense_ratio, args.aggr, args.se_threshold, args.non_iid,
                args.theta, args.attack)
        elif args.dis_check_gradient == True:
            PATH = "checkpoint/NoGradient_AckRatio{}_{}_Method{}_data{}_alpha{}_Rnd{}_Epoch{}_inject{}_dense{}_Agg{}_se_threshold{}_noniid{}_maskthreshold{}_attack{}.pt".format(
                args.num_corrupt, args.num_agents, args.method, args.data, args.alpha, rnd, args.local_ep,
                args.poison_frac, args.dense_ratio, args.aggr, args.se_threshold, args.non_iid,
                args.theta, args.attack)
        elif args.dis_pruning_aware == True:
            PATH = "checkpoint/NoPruningAware_AckRatio{}_{}_Method{}_data{}_alpha{}_Rnd{}_Epoch{}_inject{}_dense{}_Agg{}_se_threshold{}_noniid{}_maskthreshold{}_attack{}.pt".format(
                args.num_corrupt, args.num_agents, args.method, args.data, args.alpha, rnd, args.local_ep,
                args.poison_frac, args.dense_ratio, args.aggr, args.se_threshold, args.non_iid,
                args.theta, args.attack)
        
        else:
            PATH = "checkpoint/Ack{}_{}_{}_{}_alpha{}_Rnd{}_Epoch{}_inject{}_dense{}_Agg{}_sm{}_noniid{}_theta{}_attack{}_end{}_af{}_ns{}_top{}".format(
            args.num_corrupt, args.num_agents, args.method, args.data, args.alpha, rnd, args.local_ep, args.poison_frac,
            args.dense_ratio, args.aggr, args.same_mask, args.non_iid, args.theta, args.attack,
            args.cease_poison, args.anneal_factor, args.neighbour_size,args.topology)

        clean_idx= []
        for i in range(args.num_agents):
            if i not in corrupt_idx:
                clean_idx += [i]
        if  args.method == "silencer":
            torch.save({
                'option': args,
                'corrupt_model': rnd_global_params[corrupt_idx[0]],
                'clean_model': rnd_global_params[clean_idx[0]],
                'masks': [agent.mask for agent in agents],
                'malicious': corrupt_idx,
                'clean': clean_idx
            }, PATH)
        else:
            torch.save({
                'option': args,
                'corrupt_model': rnd_global_params[corrupt_idx[0]],
                'clean_model': rnd_global_params[clean_idx[0]],
                'malicious': corrupt_idx,
                'clean': clean_idx
            }, PATH)
            
            
if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True

    args = args_parser()
    
    
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    if not args.debug:
        logPath = "logs"
        if args.mask_init == "uniform":
            fileName = "uniformAck{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_dense{}_Agg{}_sm{}_noniid{}_theta{}_attack{}_af{}_ns{}_top{}".format(
                args.num_corrupt, args.num_agents, args.method, args.data, args.alpha, args.local_ep, args.poison_frac,
                args.dense_ratio, args.aggr, args.same_mask, args.non_iid, args.theta, args.attack,args.anneal_factor,args.neighbour_size,args.topology)
        elif args.dis_check_gradient == True:
            fileName = "NoGradientAck{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_dense{}_Agg{}_sm{}_noniid{}_theta{}_attack{}_af{}_ns{}_top{}".format(
                args.num_corrupt, args.num_agents, args.method, args.data, args.alpha, args.local_ep, args.poison_frac,
                args.dense_ratio, args.aggr, args.same_mask, args.non_iid, args.theta, args.attack,args.anneal_factor,args.neighbour_size,args.topology)
        elif args.noise >0:
            fileName = "Noise{}_{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_dense{}_Agg{}_sm{}_noniid{}_theta{}_attack{}_af{}_ns{}_top{}".format(args.noise,
                args.num_corrupt, args.num_agents, args.method, args.data, args.alpha, args.local_ep, args.poison_frac,
                args.dense_ratio, args.aggr, args.same_mask, args.non_iid, args.theta, args.attack,args.anneal_factor,args.neighbour_size,args.topology)
        elif args.dis_pruning_aware == True:
            fileName = "NoPruningAware{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_dense{}_Agg{}_sm{}_noniid{}_theta{}_attack{}_af{}_ns{}_top{}".format(
                args.num_corrupt, args.num_agents, args.method, args.data, args.alpha, args.local_ep, args.poison_frac,
                args.dense_ratio, args.aggr, args.same_mask, args.non_iid, args.theta, args.attack,args.anneal_factor,args.neighbour_size,args.topology)
        else:
            fileName = "Ack{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_dense{}_Agg{}_sm{}_noniid{}_theta{}_attack{}_end{}_af{}_ns{}_top{}_p{}_opt{}_s{}_per{}".format(
                args.num_corrupt, args.num_agents, args.method, args.data, args.alpha, args.local_ep, args.poison_frac,
                args.dense_ratio, args.aggr, args.same_mask, args.non_iid, args.theta, args.attack,
                args.cease_poison, args.anneal_factor, args.neighbour_size,args.topology, args.pruning, args.optimizer, args.source_class, args.personalized)
        fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName))
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)


    logging.info(args)

    cum_poison_acc_mean = 0

    # load dataset and user groups (i.e., user to data mapping)
    train_dataset, val_dataset = utils.get_datasets(args.data)
    clean_dataset = copy.deepcopy(train_dataset)
    clean_loader = DataLoader(clean_dataset, batch_size=args.bs, shuffle=True, \
                                                    num_workers=args.num_workers, pin_memory=False, drop_last=False)
    if args.data == "cifar100":
        num_target = 100
    elif args.data == "tinyimagenet":
        num_target = 200
    elif args.data == "gtsrb":
        num_target = 43
    else:
        num_target = 10
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                            pin_memory=False)
    # fedemnist is handled differently as it doesn't come with pytorch
    if args.data != 'fedemnist':
        if args.non_iid:
            user_groups = utils.distribute_data_dirichlet(train_dataset, args)
        else:
            user_groups = utils.distribute_data(train_dataset, args, n_classes=num_target)
        # print(user_groups)
    idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
    # logging.info(idxs)
    if args.data != "tinyimagenet":
        # poison the validation dataset
        poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
    else:
        poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=args)


    # construct the fp dataset by modifying the label to backdoor target
    fp_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
    for idx in idxs:
        fp_set.dataset.targets[idx] = args.target_class
    fp_loader = DataLoader(fp_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                                     pin_memory=False)
    
    poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                                     pin_memory=False)

    # poison the validation dataset
    # logging.info(idxs)
    if args.data != "tinyimagenet":
        idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
        poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        utils.poison_dataset(poisoned_val_set_only_x.dataset, args, idxs, poison_all=True, modify_label=False)
    else:
        poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=args, modify_label =False)

    poisoned_val_only_x_loader = DataLoader(poisoned_val_set_only_x, batch_size=args.bs, shuffle=False,
                                            num_workers=args.num_workers,
                                            pin_memory=False)

    # initialize a model, and the agents
    global_model = models.get_model(args.data).to(args.device)
    global_mask = {}
    neurotoxin_mask = {}
    updates_dict = {}
    n_model_params = len(parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]))
    params = {name: copy.deepcopy(global_model.state_dict()[name]) for name in global_model.state_dict()}
    if args.method == "silencer":
        sparsity = utils.calculate_sparsities(args, params, distribution=args.mask_init)
        mask = utils.init_masks(params, sparsity)
        logging.info(sparsity)
    agents, agent_data_sizes = [], {}
    corrupt_idx = np.random.choice(range(args.num_agents), args.num_corrupt, replace=False)
    logging.info("corrupt clients: {}".format(corrupt_idx))
    for _id in range(0, args.num_agents):
        if args.method == "silencer":
            if args.same_mask==0:
                agent = Agent_bi(_id, args, train_dataset, user_groups[_id], mask=utils.init_masks(params, sparsity),corrupt_idx=corrupt_idx,sparsities=sparsity)
            else:
                agent = Agent_bi(_id, args, train_dataset, user_groups[_id], mask=mask, corrupt_idx=corrupt_idx,sparsities=sparsity)
        else:
            agent = Agent(_id, args, train_dataset, user_groups[_id],corrupt_idx=corrupt_idx)
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

        # aggregation server and the loss function
    if args.personalized:
        client_per_test_loader_list,  client_per_backdoor_loader_list = construct_personalized_val(val_dataset, agents)
    criterion = nn.CrossEntropyLoss().to(args.device)
    aggregator = Aggregation(agent_data_sizes, n_model_params, poisoned_val_loader, args, None,criterion)
    agent_updates_list = []
    worker_id_list = []
    agent_updates_dict = {}

    rnd_global_params = [
        parameters_to_vector([ copy.deepcopy(global_model.state_dict()[name]) for name in global_model.state_dict()]).to("cpu") for id in range(args. num_agents)]
    before_train_params = [
        parameters_to_vector([ copy.deepcopy(global_model.state_dict()[name]) for name in global_model.state_dict()]).to("cpu") for id in range(args. num_agents)]
    
    # init wanet
    # Prepare grid
    import torch.nn.functional as F
    cross_ratio =2 
    s =0.5
    k= 4
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = (
        F.upsample(ins, size=28, mode="bicubic", align_corners=True)
        .permute(0, 2, 3, 1)
        .to(args.device)
    )
    array1d = torch.linspace(-1, 1, steps=28)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...].to(args.device)

    
    
    for rnd in range(1, args.rounds + 1):
        logging.info("--------round {} ------------".format(rnd))
        if args.topology == "full":
            # remember to clean the cache of aggregator for full topology
            aggregator.krum_update_cache = None
        if args.method == "silencer"  and  args.noise>0:
            vector_mask = [ parameters_to_vector([ agents[i].mask[name] for name in agents[i].mask])   for i in range(args.num_agents)]
        chosen = np.random.choice(args.num_agents, math.floor(args.num_agents * args.agent_frac), replace=False)
        for agent_id in chosen:
            nei_indexs =  _benefit_choose(agent_id, args.num_agents, args.neighbour_size,  topology= args.topology)
            # logging.info(torch.sum(rnd_global_params))
            if args.method == "silencer"  and  args.noise>0:
                with torch.no_grad():
                    dp_params = []
                    for i in range(args.num_agents):
                        if  i in nei_indexs:
                            dp_params += [(rnd_global_params[i].to(args.device) +  (1-vector_mask[i].to(args.device)) * (args.noise**0.5)*torch.randn(rnd_global_params[i].shape[0]).to(args.device) ).to("cpu") ]
                        else:
                            dp_params  += [rnd_global_params[i]]
                _, neurotoxin_mask, rnd_global_params[agent_id] = aggregator.aggregate_updates(   agent_id, nei_indexs, dp_params, before_train_params, global_model, rnd,agents= agents)
            else:
                import time 
                _, neurotoxin_mask, rnd_global_params[agent_id] = aggregator.aggregate_updates(   agent_id, nei_indexs, rnd_global_params, before_train_params, global_model, rnd, agents= agents)
        if args.aggr=="learn":
             # learn need to another round of aggregation on the model weights after updating
            for agent_id in chosen:
                args.aggr = "avg"
                _, neurotoxin_mask, rnd_global_params[agent_id]  = aggregator.aggregate_updates(   agent_id, nei_indexs, rnd_global_params, before_train_params, global_model, rnd, agents= agents)
                args.aggr= "learn"
        # do test after aggregation
        # logging unmask gradient norm
        
        
        do_test(rnd, global_model, rnd_global_params, args,corrupt_idx, clients=agents,identity_grid=identity_grid, noise_grid=noise_grid)
        save_checkpoints(rnd, args, agents, rnd_global_params, corrupt_idx)
        # continue training 
        unmask_gradients = {}
        for agent_id in chosen:
            state_dict = utils.vector_to_model(rnd_global_params[agent_id], global_model)
            global_model.load_state_dict(state_dict)
            global_model = global_model.to(args.device)
            update = agents[agent_id].local_train(global_model, criterion, rnd, neurotoxin_mask=neurotoxin_mask,identity_grid=identity_grid, noise_grid=noise_grid)
            before_train_params[agent_id] = copy.deepcopy(rnd_global_params[agent_id])
            rnd_global_params[agent_id] = rnd_global_params[agent_id]+ copy.deepcopy(update)

        
    # logging.info(mask_aggrement)
    logging.info('Training has finished!')
