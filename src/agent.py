import copy
import math
import time

import torch
import models
import utils
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
import torch.nn as nn
import logging
from optimizer import ESAM
import numpy as np 
class Agent():
    def __init__(self, id, args, train_dataset=None, data_idxs=None, mask=None, corrupt_idx=None):
        self.id = id
        self.args = args
        self.error = 0
        # get datasets, fedemnist is handled differently as it doesn't come with pytorch
        if train_dataset is None:
            self.train_dataset = torch.load(f'../data/Fed_EMNIST/user_trainsets/user_{id}_trainset.pt')

            # for backdoor attack, agent poisons his local dataset
            if self.id  in corrupt_idx:
                utils.poison_dataset(self.train_dataset, args, data_idxs, agent_idx=self.id)

        else:
            if self.args.data != "tinyimagenet":
                self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)
                perm = torch.randperm(len(data_idxs))
                perm = perm[: int(len(data_idxs)*0.1)]
                self.valid_dataset = utils.DatasetSplit(copy.deepcopy(train_dataset), np.array(data_idxs)[perm])
                # for backdoor attack, agent poisons his local dataset
                if self.id  in corrupt_idx:
                    self.clean_backup_dataset = copy.deepcopy(train_dataset)
                    self.data_idxs = data_idxs
                    utils.poison_dataset(train_dataset, args, data_idxs, agent_idx=self.id)
            else:
                self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs, runtime_poison=True, args=args,
                                                        client_id=id)
        
        
        # get dataloader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                       num_workers=args.num_workers, pin_memory=False, drop_last=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.args.bs, shuffle=True, \
                                       num_workers=args.num_workers, pin_memory=False, drop_last=True)
        self.train_loader_bs1 = DataLoader(self.train_dataset, batch_size=1, shuffle=False, \
                                       num_workers=args.num_workers, pin_memory=False, drop_last=True)
        self.train_loader_bs2 = DataLoader(self.train_dataset, batch_size=1, shuffle=False, \
                                           num_workers=args.num_workers, pin_memory=False, drop_last=True)
        self.corrupt_idx=corrupt_idx
        # size of local dataset
        self.n_data = len(self.train_dataset)
        
        # print(corrupt_idx)

    def screen_gradients(self, model):
        model.eval()
        # # # train and update
        criterion = nn.CrossEntropyLoss()
        gradient = {name: 0 for name, param in model.named_parameters()}

        batch_num = 0
        correctly_labeled_samples=0
        norm=0
        for _, (x, labels) in enumerate(self.train_loader_bs1):
            batch_num+=1
            model.zero_grad()
            x, labels = x.to(self.args.device), labels.to(self.args.device)
            log_probs = model.forward(x)
            _, pred_labels = torch.max(log_probs, 1)
            pred_labels = pred_labels.view(-1)
            minibatch_loss = criterion(log_probs, labels.long())
            loss = minibatch_loss
            loss.backward()
            correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()
            for name, param in model.named_parameters():
                gradient[name] += torch.square(param.grad.data)
                norm +=torch.norm(param.grad.data)
            break
        # accuracy = correctly_labeled_samples / len(self.train_loader)
        # print(norm/ len(self.train_loader))
        # print(loss)
        return gradient

    def screen_hessian(self, model):
        import functorch
        names = list(n for n, _ in model.named_parameters())
        model.eval()
        # # # train and update
        criterion = nn.CrossEntropyLoss()
        num_param = sum(p.numel() for p in model.parameters())
        names = list(n for n, _ in model.named_parameters())
        diagonal = 0
        for _, (x, labels) in enumerate(self.train_loader_bs2):
            model.zero_grad()
            x, labels = x.to(self.args.device), labels.to(self.args.device)
            func, func_params,buffers  = functorch.make_functional_with_buffers(model)  # New
            def loss(params):
                out = func(params, x)
                return criterion(out, labels)
            print(functorch.hessian (loss)(func_params))
            break
            # get_esd_plot(density_eigen, density_weight)
        # print(diagonal)
        return diagonal

    def check_poison_timing(self, round):
        if round > self.args.cease_poison:
            self.train_dataset = utils.DatasetSplit(self.clean_backup_dataset, self.data_idxs)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                           num_workers=self.args.num_workers, pin_memory=False, drop_last=True)

    def local_train(self, global_model, criterion, round=None, neurotoxin_mask=None, identity_grid=None, noise_grid=None):
        """ Do a local training over the received global model, return the update """
        initial_global_model_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        # if self.id in self.corrupt_idx:
        #     self.check_poison_timing(round)
        global_model.train()
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr * (self.args.lr_decay) ** round,
                                    weight_decay=self.args.wd)
        
        start_time = time.time()
        for _ in range(self.args.local_ep):
            start = time.time()
            for _, (inputs, labels) in enumerate(self.train_loader):
                if self.args.optimizer =="sam":
                    base_optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr * (self.args.lr_decay) ** round,
                                    weight_decay=self.args.wd)
                    optimizer = ESAM(global_model.parameters(), base_optimizer)
                    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
                    inputs, labels = inputs.to(device=self.args.device, non_blocking=True), \
                                    labels.to(device=self.args.device, non_blocking=True)
                    def defined_backward(loss):
                        loss.backward()
                    paras = [inputs, labels, loss_fn, global_model, defined_backward]
                    optimizer.paras = paras
                    optimizer.step()
                else:
                    
                    
                    optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr * (self.args.lr_decay) ** round,
                                    weight_decay=self.args.wd)
                    optimizer.zero_grad()
                    inputs, labels = inputs.to(device=self.args.device, non_blocking=True), \
                                    labels.to(device=self.args.device, non_blocking=True)
                                    
                                    
                    if self.args.attack == "wanet":
                        import torch.nn.functional as F
                        cross_ratio =0.5 
                        s =0.5
                        k= 4
                        bs = inputs.shape[0]
                         # Create backdoor data
                        num_bd = int(bs * self.args.poison_frac)
                        num_cross = int(num_bd * cross_ratio)
                        grid_temps = (identity_grid + s * noise_grid / 28) 
                        grid_temps = torch.clamp(grid_temps, -1, 1)
                        ins = torch.rand(num_cross, 28, 28, 2).to(self.args.device) * 2 - 1
                        grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / 28
                        grid_temps2 = torch.clamp(grid_temps2, -1, 1)
                        inputs_bd = F.grid_sample(inputs[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
                        inputs_cross = F.grid_sample(inputs[num_bd : (num_bd + num_cross)], grid_temps2, align_corners=True)
                        targets_bd = torch.ones_like(labels[:num_bd]) * self.args.target_class
                        inputs = torch.cat([inputs_bd, inputs_cross, inputs[(num_bd + num_cross) :]], dim=0)
                        labels = torch.cat([targets_bd, labels[num_bd:]], dim=0)
                    outputs = global_model(inputs)
                    minibatch_loss = criterion(outputs, labels)
                    minibatch_loss.backward()
                    if self.args.attack == "neurotoxin" and len(neurotoxin_mask) and   self.id in self.corrupt_idx:
                        for name, param in global_model.named_parameters():
                            param.grad.data = neurotoxin_mask[name].to(self.args.device) * param.grad.data
                    if self.args.attack == "r_neurotoxin" and len(neurotoxin_mask) and   self.id in self.corrupt_idx:
                        for name, param in global_model.named_parameters():
                            param.grad.data = (torch.ones_like(neurotoxin_mask[name].to(self.args.device))-neurotoxin_mask[name].to(self.args.device) ) * param.grad.data

                    torch.nn.utils.clip_grad_norm_(global_model.parameters(), 10)
                    optimizer.step()


            
            # logging.info(end - start)
        end_time = time.time()
        local_training_time = end_time - start_time
        logging.info("local training time{}".format(local_training_time))
        with torch.no_grad():
            after_train = parameters_to_vector(
                [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
            self.update = after_train - initial_global_model_params
            return self.update.to("cpu")