import torch
from torch.nn.utils import vector_to_parameters, parameters_to_vector
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


