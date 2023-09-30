import copy

import torch
import models
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import numpy as np
from copy import deepcopy
from torch.nn import functional as F
import logging
from utils import vector_to_model

class Aggregation():
    def __init__(self, agent_data_sizes, n_params, poisoned_val_loader, args, writer):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.writer = writer
        self.server_lr = args.server_lr
        self.n_params = n_params
        self.poisoned_val_loader = poisoned_val_loader
        self.cum_net_mov = 0
        self.krum_update_cache = None
        
    
    
    def aggregate_updates(self, client_id,  nei_indexs, after_train_params,before_train_params, global_model, round):
        # logging.info(nei_indexs)
        if self.args.aggr =="avg":
            weight_dict ={}
            for _id, weight in enumerate(after_train_params):
                if _id in nei_indexs or _id == client_id:
                    weight_dict[_id] = weight
            average_weights =   self.decentralized_avg(weight_dict)
        elif self.args.aggr =="krum" or self.args.aggr =="rlr" or self.args.aggr =="fltrust":
            if round>1:
                updates = {}
                for _id, weight in enumerate(after_train_params):
                    if _id in nei_indexs or _id == client_id:
                        updates[_id] = weight  - before_train_params[client_id]
        if self.args.aggr =="krum":
            if self.args.topology !="full":
                krum_update = self.agg_krum(updates)
            else:
                # for full topology, we reuse the update compute by other clients to save computation
                if self.krum_update_cache==None:
                    self.krum_update_cache = self.agg_krum(updates)
                    krum_update= self.krum_update_cache 
                else:
                    krum_update= self.krum_update_cache
            average_weights = krum_update+before_train_params[client_id]
        elif self.args.aggr == "rlr":
            # logging.info(updates)
            self.args.theta = len(updates)*0.5
            lr, _= self.compute_robustLR(updates)
            average_update =   self.decentralized_avg(updates )
            average_weights = lr* average_update+before_train_params[client_id]
        elif self.args.aggr == "fltrust":
            # logging.info(updates)
            if round>1:
                # logging.info("fuck")
                average_update= self.compute_robusttrust(updates, client_id )
                average_weights = average_update+before_train_params[client_id]
            else:
                # logging.info("hi")
                weight_dict ={}
                for _id, weight in enumerate(after_train_params):
                    if _id in nei_indexs or _id == client_id:
                        weight_dict[_id] = weight
                average_weights =   self.decentralized_avg(weight_dict)
        average_update = average_weights-before_train_params[client_id]
        neurotoxin_mask = {}
        updates_dict = vector_to_model(average_update, global_model)
        for name in updates_dict:
            updates = updates_dict[name].abs().view(-1)
            gradients_length = torch.numel(updates)
            _, indices = torch.topk(-1 * updates, int(gradients_length * self.args.dense_ratio))
            mask_flat = torch.zeros(gradients_length)
            mask_flat[indices.cpu()] = 1
            neurotoxin_mask[name] = (mask_flat.reshape(updates_dict[name].size()))
        return    None, neurotoxin_mask, average_weights
    
    def compute_robusttrust(self, agent_updates, id):
        total_TS = 0
        TSnorm = {}
        for key in agent_updates:
            if id!=key:
                update = agent_updates[key]
                TS = torch.dot(update,agent_updates[id])/(torch.norm(update)*torch.norm(agent_updates[id]))
                if TS < 0:
                    TS = 0
                total_TS += TS
                # logging.info(TS)
                norm = torch.norm(agent_updates[id])/torch.norm(update)
                TSnorm[key] = TS*norm
        average_update =  0
        
        for key in agent_updates :
            if id!=key:
                average_update += TSnorm[key]*agent_updates[key]
        average_update /= (total_TS + 1e-6)
        return average_update
    
    def compute_robustLR(self, agent_updates):

        agent_updates_sign = [torch.sign(update) for update in agent_updates.values()]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        mask=torch.zeros_like(sm_of_signs)
        mask[sm_of_signs < self.args.theta] = 0
        mask[sm_of_signs >= self.args.theta] = 1
        sm_of_signs[sm_of_signs < self.args.theta] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.theta] = self.server_lr
        return sm_of_signs, mask

    def decentralized_avg(self,  agent_updates_dict):
        # Use the received models to infer the consensus model
        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates += n_agent_data * update
            total_data += n_agent_data
        return sm_updates / total_data
    
    def agg_krum(self, agent_updates_dict):
        krum_param_m = 1
        def _compute_krum_score( agent_updates_list, byzantine_client_num):
            with torch.no_grad():
                krum_scores = []
                num_client = len(agent_updates_list)
                # logging.info(num_client)
                for i in range(0, num_client):
                    dists = []
                    for j in range(0, num_client):
                        if i != j:
                            dists.append(
                                torch.norm(agent_updates_list[i].to(self.args.device)- agent_updates_list[j].to(self.args.device))
                                .item() ** 2
                            )
                    dists.sort()  # ascending
                    score = dists[0: num_client - byzantine_client_num - 2]
                    krum_scores.append(sum(score))
            # logging.info("finish")
            return krum_scores

        # Compute list of scores
        agent_updates_list = list(agent_updates_dict.values())
        byzantine_num = min( self.args.num_corrupt, len(agent_updates_dict)-1) 
        krum_scores = _compute_krum_score(agent_updates_list, byzantine_num)
        score_index = torch.argsort(
            torch.Tensor(krum_scores)
        ).tolist()  # indices; ascending
        score_index = score_index[0: krum_param_m]
        return_gradient = [agent_updates_list[i] for i in score_index]
        return (sum(return_gradient)/len(return_gradient)).to("cpu")
    
    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """

        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates +=   n_agent_data *  update
            total_data += n_agent_data
        return  sm_updates / total_data
    
    def agg_comed(self, agent_updates_dict):
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        return torch.median(concat_col_vectors, dim=1).values
    
    def agg_sign(self, agent_updates_dict):
        """ aggregated majority sign update """
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_signs = torch.sign(sum(agent_updates_sign))
        return torch.sign(sm_signs)


    def clip_updates(self, agent_updates_dict):
        for update in agent_updates_dict.values():
            l2_update = torch.norm(update, p=2) 
            update.div_(max(1, l2_update/self.args.clip))
        return
                  
    def plot_norms(self, agent_updates_dict, cur_round, norm=2):
        """ Plotting average norm information for honest/corrupt updates """
        honest_updates, corrupt_updates = [], []
        for key in agent_updates_dict.keys():
            if key < self.args.num_corrupt:
                corrupt_updates.append(agent_updates_dict[key])
            else:
                honest_updates.append(agent_updates_dict[key])
                              
        l2_honest_updates = [torch.norm(update, p=norm) for update in honest_updates]
        avg_l2_honest_updates = sum(l2_honest_updates) / len(l2_honest_updates)
        self.writer.add_scalar(f'Norms/Avg_Honest_L{norm}', avg_l2_honest_updates, cur_round)
        
        if len(corrupt_updates) > 0:
            l2_corrupt_updates = [torch.norm(update, p=norm) for update in corrupt_updates]
            avg_l2_corrupt_updates = sum(l2_corrupt_updates) / len(l2_corrupt_updates)
            self.writer.add_scalar(f'Norms/Avg_Corrupt_L{norm}', avg_l2_corrupt_updates, cur_round) 
        return
        
    def comp_diag_fisher(self, model_params, data_loader, adv=True):

        model = models.get_model(self.args.data)
        vector_to_parameters(model_params, model.parameters())
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        precision_matrices = {}
        for n, p in deepcopy(params).items():
            p.data.zero_()
            precision_matrices[n] = p.data
            
        model.eval()
        for _, (inputs, labels) in enumerate(data_loader):
            model.zero_grad()
            inputs, labels = inputs.to(device=self.args.device, non_blocking=True),\
                                    labels.to(device=self.args.device, non_blocking=True).view(-1, 1)
            if not adv:
                labels.fill_(self.args.base_class)
                
            outputs = model(inputs)
            log_all_probs = F.log_softmax(outputs, dim=1)
            target_log_probs = outputs.gather(1, labels)
            batch_target_log_probs = target_log_probs.sum()
            batch_target_log_probs.backward()
            
            for n, p in model.named_parameters():
                precision_matrices[n].data += (p.grad.data ** 2) / len(data_loader.dataset)
                
        return parameters_to_vector(precision_matrices.values()).detach()

        
    def plot_sign_agreement(self, robustLR, cur_global_params, new_global_params, cur_round):
        """ Getting sign agreement of updates between honest and corrupt agents """
        # total update for this round
        update = new_global_params - cur_global_params
        
        # compute FIM to quantify these parameters: (i) parameters which induces adversarial mapping on trojaned, (ii) parameters which induces correct mapping on trojaned
        fisher_adv = self.comp_diag_fisher(cur_global_params, self.poisoned_val_loader)
        fisher_hon = self.comp_diag_fisher(cur_global_params, self.poisoned_val_loader, adv=False)
        _, adv_idxs = fisher_adv.sort()
        _, hon_idxs = fisher_hon.sort()
        
        # get most important n_idxs params
        n_idxs = self.args.top_frac #math.floor(self.n_params*self.args.top_frac)
        adv_top_idxs = adv_idxs[-n_idxs:].cpu().detach().numpy()
        hon_top_idxs = hon_idxs[-n_idxs:].cpu().detach().numpy()
        
        # minimized and maximized indexes
        min_idxs = (robustLR == -self.args.server_lr).nonzero().cpu().detach().numpy()
        max_idxs = (robustLR == self.args.server_lr).nonzero().cpu().detach().numpy()
        
        # get minimized and maximized idxs for adversary and honest
        max_adv_idxs = np.intersect1d(adv_top_idxs, max_idxs)
        max_hon_idxs = np.intersect1d(hon_top_idxs, max_idxs)
        min_adv_idxs = np.intersect1d(adv_top_idxs, min_idxs)
        min_hon_idxs = np.intersect1d(hon_top_idxs, min_idxs)
       
        # get differences
        max_adv_only_idxs = np.setdiff1d(max_adv_idxs, max_hon_idxs)
        max_hon_only_idxs = np.setdiff1d(max_hon_idxs, max_adv_idxs)
        min_adv_only_idxs = np.setdiff1d(min_adv_idxs, min_hon_idxs)
        min_hon_only_idxs = np.setdiff1d(min_hon_idxs, min_adv_idxs)
        
        # get actual update values and compute L2 norm
        max_adv_only_upd = update[max_adv_only_idxs] # S1
        max_hon_only_upd = update[max_hon_only_idxs] # S2
        
        min_adv_only_upd = update[min_adv_only_idxs] # S3
        min_hon_only_upd = update[min_hon_only_idxs] # S4


        #log l2 of updates
        max_adv_only_upd_l2 = torch.norm(max_adv_only_upd).item()
        max_hon_only_upd_l2 = torch.norm(max_hon_only_upd).item()
        min_adv_only_upd_l2 = torch.norm(min_adv_only_upd).item()
        min_hon_only_upd_l2 = torch.norm(min_hon_only_upd).item()
       
        self.writer.add_scalar(f'Sign/Hon_Maxim_L2', max_hon_only_upd_l2, cur_round)
        self.writer.add_scalar(f'Sign/Adv_Maxim_L2', max_adv_only_upd_l2, cur_round)
        self.writer.add_scalar(f'Sign/Adv_Minim_L2', min_adv_only_upd_l2, cur_round)
        self.writer.add_scalar(f'Sign/Hon_Minim_L2', min_hon_only_upd_l2, cur_round)
        
        
        net_adv =  max_adv_only_upd_l2 - min_adv_only_upd_l2
        net_hon =  max_hon_only_upd_l2 - min_hon_only_upd_l2
        self.writer.add_scalar(f'Sign/Adv_Net_L2', net_adv, cur_round)
        self.writer.add_scalar(f'Sign/Hon_Net_L2', net_hon, cur_round)
        
        self.cum_net_mov += (net_hon - net_adv)
        self.writer.add_scalar(f'Sign/Model_Net_L2_Cumulative', self.cum_net_mov, cur_round)
        return

