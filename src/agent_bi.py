import copy
import math
import time

import torch
import models
import utils
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import logging
from optimizer import ESAM


# This is the agent class for silencer
class Agent():
    def __init__(self, id, args, train_dataset=None, data_idxs=None, mask=None, sparsities=None, corrupt_idx=None):
        self.id = id
        self.args = args
        self.error = 0
        if self.args.data != "tinyimagenet":
            self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)
            # for backdoor attack, agent poisons his local dataset
            if self.id in corrupt_idx:
                # self.clean_backup_dataset = copy.deepcopy(train_dataset)
                self.data_idxs = data_idxs
                self. clean_dataset = copy.deepcopy(self.train_dataset)
                utils.poison_dataset(train_dataset, args, data_idxs, agent_idx=self.id)
                
        else:
            self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs, runtime_poison=True, args=args,
                                                        client_id=id)

        # get dataloader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                       num_workers=args.num_workers, pin_memory=False, drop_last=True)
        
        # get dataloader
        if self.id in corrupt_idx:
            self.clean_data_loader = DataLoader(self.clean_dataset, batch_size=self.args.bs, shuffle=True, \
                                        num_workers=args.num_workers, pin_memory=False, drop_last=True)
        # size of local dataset
        self.n_data = len(self.train_dataset)
        self.corrupt_idx=corrupt_idx
        # all cients are initialized the same mask
        self.mask = copy.deepcopy(mask)
        self.sparsities = sparsities
        self.num_remove= None



    def screen_FI(self, model, loader):
        # also return the gradient to check gradient norm (required by one reviewer)
        model.train()
        # # # train and update
        criterion = nn.CrossEntropyLoss()
        FI = {name: 0 for name, param in model.named_parameters()}
        gradient =  {name: 0 for name, param in model.named_parameters()}
        for _, (x, labels) in enumerate(loader):
            model.zero_grad()
            x, labels = x.to(self.args.device), labels.to(self.args.device)
            log_probs = model.forward(x)
            minibatch_loss = criterion(log_probs, labels.long())
            loss = minibatch_loss
            loss.backward()
            # when applying cross entropy, digonal of empirical Fisher information is the square of gradient
            for name, param in model.named_parameters():
                FI[name] += torch.square(param.grad.data)
                gradient [name] += param.grad.data
        return FI, gradient

    def update_mask(self, masks, num_remove, FI=None):
        for name in FI:
            if self.args.dis_check_gradient:
                if num_remove[name]>0:
                    temp = torch.where(masks[name] == 0, torch.ones_like(masks[name]), torch.zeros_like(masks[name]))
                    idx = torch.multinomial(temp.flatten().to(self.args.device), num_remove[name], replacement=False)
                    masks[name].view(-1)[idx] = 1
            else:
                temp = torch.where(masks[name].to(self.args.device) == 0, torch.abs(FI[name]),
                                    -100000 * torch.ones_like(FI[name]))
                sort_temp, idx = torch.sort(temp.view(-1), descending=True)
                masks[name].view(-1)[idx[:num_remove[name]]] = 1
        return masks

             

    def fire_mask(self, model, round, FI):
      
        drop_ratio = self.args.anneal_factor / 2 * (1 + np.cos((round * np.pi) / (self.args.rounds)))
        
        # logging.info(drop_ratio)
        masks =self.mask
        num_remove = {}
        for name in masks:
            if self.sparsities[name]>0:
                num_non_zeros = torch.sum(masks[name].to(self.args.device))
                num_remove[name] = math.floor(drop_ratio * num_non_zeros)
            else:
                num_remove[name] = 0
        for name in masks:
            if num_remove[name]>0 and  "track" not in name and "running" not in name: 
                if self.args.pruning=="random":
                    temp = torch.where(masks[name] == 1, torch.ones_like(masks[name]), torch.zeros_like(masks[name]))
                    idx = torch.multinomial(temp.flatten().to(self.args.device), num_remove[name], replacement=False)
                    masks[name].view(-1)[idx] = 0
                elif self.args.pruning=="FI":
                    temp_weights = torch.where(masks[name].to(self.args.device) > 0,  FI[name]  ,
                                            100000 * torch.ones_like(FI[name]))
                    x, idx = torch.sort(temp_weights.view(-1).to(self.args.device))
                    masks[name].view(-1)[idx[:num_remove[name]]] = 0
                else:
                    temp_weights = torch.where(masks[name].to(self.args.device) > 0,  torch.abs(model.state_dict()[name]   ),
                                            100000 * torch.ones_like(FI[name]))
                    x, idx = torch.sort(temp_weights.view(-1).to(self.args.device))
                    masks[name].view(-1)[idx[:num_remove[name]]] = 0
        return masks, num_remove

    def local_train(self, global_model, criterion, round=None, neurotoxin_mask =None,identity_grid=None, noise_grid=None):
        """ Do a local training over the received global model, return the update """
        initial_global_model_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        global_model.to(self.args.device)
        for name, param in global_model.named_parameters():
            self.mask[name] =self.mask[name].to(self.args.device)
            param.data = param.data * self.mask[name]
        start_time = time.time()
        if (self.args.attack !="benign_search" )  or self.id not in self.corrupt_idx:
            FI,unmask_gradient = self.screen_FI(global_model, loader = self.train_loader)
        else:
            FI,unmask_gradient = self.screen_FI(global_model, loader = self.clean_data_loader)
        end_time = time.time()
        Screen_FI = end_time - start_time
        # logging.info("screen FI time{}".format(Screen_FI))
        if (self.args.attack !="fix_mask" )  or self.id not in self.corrupt_idx:
            self.mask, self.num_remove = self.fire_mask(global_model, round, FI)
            for name, param in global_model.named_parameters():
                self.mask[name] =self.mask[name].to(self.args.device)
                param.data = param.data * self.mask[name]  
            if self.num_remove!=None:
                if self.id not in  self.corrupt_idx or self.args.attack!="fix_mask" :
                    self.mask = self.update_mask(self.mask, self.num_remove, FI)
        end_time2 = time.time()
        searching_time = end_time2 - end_time
        # logging.info("Mask searching time{}".format(searching_time))
        global_model.train()
        lr = self.args.client_lr* (self.args.lr_decay)**round
        optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, weight_decay=self.args.wd)

        for _ in range(self.args.local_ep):
            for _, (inputs, labels) in enumerate(self.train_loader):
                
                if self.args.optimizer =="sam":
                    # logging.info("i am sam")
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
                    for name, param in global_model.named_parameters():
                        param.data = self.mask[name].to(self.args.device) * param.data
                else:
                    optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr * (self.args.lr_decay) ** round,
                                    weight_decay=self.args.wd)
                    optimizer.zero_grad()
                    inputs, labels = inputs.to(device=self.args.device), \
                                    labels.to(device=self.args.device)
                    outputs = global_model(inputs)
                    minibatch_loss = criterion(outputs, labels)
                    loss = minibatch_loss
                    loss.backward()
                    if self.args.attack == "neurotoxin" and len(neurotoxin_mask) and   self.id in self.corrupt_idx:
                        for name, param in global_model.named_parameters():
                            param.grad.data = neurotoxin_mask[name].to(self.args.device) * param.grad.data
                    for name, param in global_model.named_parameters():
                        param.grad.data = self.mask[name].to(self.args.device) * param.grad.data
        
                    optimizer.step()
        
        end_time3 = time.time()
        local_training_time = end_time3 - end_time2
        # logging.info("local training time{}".format(local_training_time))
        with torch.no_grad():
            after_train = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
            array_mask = parameters_to_vector([ self.mask[name].to(self.args.device) for name in global_model.state_dict()]).detach()
            self.update = ( array_mask *(after_train - initial_global_model_params))
            if "scale" in self.args.attack:
                # logging.info("scale update for" + self.args.attack.split("_",1)[1] + " times")
                if self.id<  self.args.num_corrupt:
                    self.update=  int(self.args.attack.split("_",1)[1]) * self.update
                    
        
        return self.update.to("cpu")
