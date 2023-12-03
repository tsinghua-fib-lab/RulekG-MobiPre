# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Dict

import math
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import datetime
from datetime import datetime
import torch.nn.functional as F
import time
class TKBCModel(nn.Module, ABC):

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass
   
    
    def get_mrr(self,target, scores):
        ranks = []
        sort_values, sort_idxs = torch.sort(scores, dim=1, descending=True)
        sort_idxs = sort_idxs.cpu().numpy()
        for j in range(scores.shape[0]):
            rank = np.where(sort_idxs[j] == target[j].item())[0][0]

            ranks.append(rank+1)
        return ranks

       
    def get_ranking(
            self, queries: torch.Tensor, split,
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        start = time.time()
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.zeros(len(queries))
        with torch.no_grad():
            c_begin = 0
            user_rank = []
            while c_begin < self.sizes[2]:
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    scores = self.get_queries(these_queries)

                    target = these_queries[:,3]
                    user_rank += these_queries[:,0].tolist()
                    rank = self.get_mrr(target, scores)
                    ranks[b_begin:b_begin + batch_size] += torch.FloatTensor(rank)

                    b_begin += batch_size
                c_begin += chunk_size
        print(time.time()-start)
        return ranks, user_rank

# self.n_ids, self.n_timestamps, self.n_pois, self.n_access,  self.n_deltat, self.n_inverse, self.n_transfer
# [users_to_id[lhs], timestamps_to_id[ts], access_to_id['access'],pois_to_id[rhs], timestamps_to_id[ots], inverse_to_id['inverse'],pois_to_id[origin],int(deltat)]

class ComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int, int, int, int, int], rank: int,
             test_transition=True, init_size: float = 1e-3
    ):
        
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank
       
        self.time_nl = torch.sin
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank)
            for s in [sizes[2], sizes[6], sizes[2], sizes[5], sizes[3], sizes[4], sizes[4], sizes[4], sizes[1], sizes[1],sizes[0],sizes[7]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size
        self.embeddings[4].weight.data *= init_size
        self.embeddings[5].weight.data *= init_size
        self.embeddings[6].weight.data *= init_size
        self.embeddings[7].weight.data *= init_size
        self.embeddings[8].weight.data *= init_size
        self.embeddings[9].weight.data *= init_size
        self.embeddings[10].weight.data *= init_size
        self.embeddings[11].weight.data *= init_size
        self.bn = nn.BatchNorm1d(sizes[2])
        self.hidden1 = torch.nn.Linear(5*rank, 200)
        self.hidden2 = torch.nn.Linear(3*rank, 200)
        self.out = torch.nn.Linear(200, rank)
        self.test_transition = test_transition

    @staticmethod
    def has_time():
        return True

    def diachronic(self, x):
        delta_a = self.embeddings[5](x)
        delta_freq = self.embeddings[6](x)
        delta_phi = self.embeddings[7](x)
        delta = delta_a * self.time_nl(delta_freq * x.type(torch.FloatTensor).cuda().reshape((-1,1))  + delta_phi)
        return delta

    def forward(self, x, path2_origin, path2_mid, path2_user, similar):
        origin = self.embeddings[0](x[:, 6])
        tran = self.embeddings[1](x[:, 8])
        loc = self.embeddings[2](x[:, 3])
        tim = self.embeddings[9](x[:, 1])
        inv = self.embeddings[3](x[:, 5])
        right = self.embeddings[2].weight
        acc = self.embeddings[4](x[:, 2])
        delta = self.diachronic(x[:, 7])
        tim1 = self.embeddings[8](x[:, 4])
 
        
  
        origin = origin[:, :self.rank], origin[:, self.rank:]
        loc = loc[:, :self.rank], loc[:, self.rank:]
        tran = tran[:, :self.rank], tran[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        inv = inv[:, :self.rank], inv[:, self.rank:]
        right = right[:, :self.rank], right[:, self.rank:]
        acc = acc[:, :self.rank], acc[:, self.rank:]
        delta = delta[:, :self.rank], delta[:, self.rank:]
        tim1 = tim1[:, :self.rank], tim1[:, self.rank:]



        path1 = torch.cat([acc[0],tim1[0],delta[0],tim[0],inv[0]],1),torch.cat([acc[1],tim1[1],delta[1],tim[1],inv[1]],1)
        rel_path1 = self.out(F.relu(self.hidden1(path1[0]))),self.out(F.relu(self.hidden1(path1[1])))
        eng1 = rel_path1[0] - tran[0], rel_path1[1] - tran[1]


        path2_origin = self.embeddings[0](path2_origin)
        path2_mid = self.embeddings[0](path2_mid)
        path2_tran  = self.embeddings[1](x[:,8][0].repeat(len(path2_origin)))
        # path2_right  = self.embeddings[2].weight

        path2_origin = path2_origin[:, :self.rank], path2_origin[:, self.rank:]
        path2_mid = path2_mid[:, :self.rank], path2_mid[:, self.rank:]
        path2_tran = path2_tran[:, :self.rank], path2_tran[:, self.rank:]
        # path2_right = path2_right[:,:self.rank], path2_right[:, self.rank:]

        path2 = torch.cat([path2_tran[0],path2_tran[0],path2_mid[0]],1),torch.cat([path2_tran[1],path2_tran[1],path2_mid[1]],1)
        rel_path2 = self.out(F.relu(self.hidden2(path2[0]))),self.out(F.relu(self.hidden2(path2[1])))
        eng2 = rel_path2[0] - path2_tran[0], rel_path2[1] - path2_tran[1]

        user1 = self.embeddings[10](similar[:, 0])
        user1 = nn.functional.normalize(user1,dim=1)
        sim = self.embeddings[11](similar[:, 1])
        right_sim = self.embeddings[10].weight
        user1 = user1[:, :self.rank], user1[:, self.rank:]
        sim = sim[:, :self.rank], sim[:, self.rank:]
        right_sim = right_sim[:, :self.rank], right_sim[:, self.rank:]

        sim_score = (user1[0] * sim[0] - user1[1] * sim[1]) @ right_sim[0].t() + (user1[1] * sim[0] + user1[0] * sim[1]) @ right_sim[1].t()

        if not self.test_transition:
            score = (tim[0] * inv[0] - tim[1] * inv[1]) @ right[0].t() + (tim[1] * inv[0] + tim[0] * inv[1]) @ right[1].t()
        else:
            score = (origin[0] * tran[0] - origin[1] * tran[1]) @ right[0].t() + (origin[1] * tran[0] + origin[0] * tran[1]) @ right[1].t()
        
        return score, sim_score, ((origin[0] * rel_path1[0] - origin[1] * rel_path1[1]) @ right[0].t() + (origin[1] * rel_path1[0] + origin[0] * rel_path1[1]) @ right[1].t()),(
 (path2_origin[0] * rel_path2[0] - path2_origin[1] * rel_path2[1]) @ right[0].t() + (path2_origin[1] * rel_path2[0] + path2_origin[0] * rel_path2[1]) @ right[1].t()),(
                   torch.sqrt(origin[0] ** 2 + origin[1] ** 2),
                   torch.sqrt(tran[0] ** 2 + tran[1] ** 2),
                   torch.sqrt(loc[0] ** 2 + loc[1] ** 2),
        )
               

    def forward1(self, x):
        origin = self.embeddings[0](x[:, 6])
        loc = self.embeddings[2](x[:, 3])
        inv = self.embeddings[3](x[:, 5])
        acc = self.embeddings[4](x[:, 2])
        delta = self.diachronic(x[:, 7])
        tim1 = self.embeddings[8](x[:, 4])
        tim = self.embeddings[9](x[:, 1])
        
        right_ots = self.embeddings[8].weight
        right_ts = self.embeddings[9].weight
        right_loc = self.embeddings[2].weight

        origin = origin[:, :self.rank], origin[:, self.rank:]
        loc = loc[:, :self.rank], loc[:, self.rank:]
        delta = delta[:, :self.rank], delta[:, self.rank:]
        inv = inv[:, :self.rank], inv[:, self.rank:]
        acc = acc[:, :self.rank], acc[:, self.rank:]
        
        tim1 = tim1[:, :self.rank], tim1[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        right_ots = right_ots[:, :self.rank], right_ots[:, self.rank:]
        right_ts = right_ts[:, :self.rank], right_ts[:, self.rank:]
        right_loc = right_loc[:, :self.rank], right_loc[:, self.rank:]

        return ((origin[0] * acc[0] - origin[1] * acc[1]) @ right_ots[0].t() + (origin[1] * acc[0] + origin[0] * acc[1]) @ right_ots[1].t() ),(
            (tim1[0] * delta[0] - tim1[1] * delta[1]) @ right_ts[0].t() + (tim1[1] * delta[0] + tim1[0] * delta[1]) @ right_ts[1].t()),(
              (tim[0] * inv[0] - tim[1] * inv[1]) @ right_loc[0].t() + (tim[1] * inv[0] + tim[0] * inv[1]) @ right_loc[1].t() )  

    def get_queries(self, queries: torch.Tensor):
        origin = self.embeddings[0](queries[:, 6])
        tran = self.embeddings[1](queries[:, 8])
        loc = self.embeddings[2](queries[:, 3])
        tim = self.embeddings[9](queries[:, 1])
        inv = self.embeddings[3](queries[:, 5])
        right = self.embeddings[2].weight
        
  
        origin = origin[:, :self.rank], origin[:, self.rank:]
        loc = loc[:, :self.rank], loc[:, self.rank:]
        tran = tran[:, :self.rank], tran[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        inv = inv[:, :self.rank], inv[:, self.rank:]
        right = right[:, :self.rank], right[:, self.rank:]

        if not self.test_transition:
            score = (tim[0] * inv[0] - tim[1] * inv[1]) @ right[0].t() + (tim[1] * inv[0] + tim[0] * inv[1]) @ right[1].t()
        else:
            score = (origin[0] * tran[0] - origin[1] * tran[1]) @ right[0].t() + (origin[1] * tran[0] + origin[0] * tran[1]) @ right[1].t()
        return  score

class RComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int, int, int, int, int], rank: int,
            user_ratio: float, test_transition=True, init_size: float = 1e-3
    ):
        
        super(RComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.user_ratio = user_ratio
       
        self.time_nl = torch.sin
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank)
            for s in [sizes[2], sizes[6], sizes[2], sizes[5], sizes[3], sizes[4], sizes[4], sizes[4], sizes[1], sizes[1],sizes[0],sizes[7]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size
        self.embeddings[4].weight.data *= init_size
        self.embeddings[5].weight.data *= init_size
        self.embeddings[6].weight.data *= init_size
        self.embeddings[7].weight.data *= init_size
        self.embeddings[8].weight.data *= init_size
        self.embeddings[9].weight.data *= init_size
        self.embeddings[10].weight.data *= init_size
        self.embeddings[11].weight.data *= init_size
        self.bn = nn.BatchNorm1d(sizes[2])
        self.test_transition = test_transition

    @staticmethod
    def has_time():
        return True

    def diachronic(self, x):
        delta_a = self.embeddings[5](x)
        delta_freq = self.embeddings[6](x)
        delta_phi = self.embeddings[7](x)
        delta = delta_a * self.time_nl(delta_freq * x.type(torch.FloatTensor).cuda().reshape((-1,1))  + delta_phi)
        return delta

    def forward(self, x,similar):
        origin = self.embeddings[0](x[:, 6])
        tran = self.embeddings[1](x[:, 8])
        loc = self.embeddings[2](x[:, 3])
        right = self.embeddings[2].weight
        trand = tran[:, :int(self.user_ratio * self.rank)], tran[:, self.rank:self.rank + int(self.user_ratio * self.rank)]
        trans = tran[:, int(self.user_ratio * self.rank):self.rank], tran[:, self.rank + int(self.user_ratio * self.rank):]
        tran = tuple(torch.cat((x, y), 1) for x, y in zip(trand, trans))
  
        origin = origin[:, :self.rank], origin[:, self.rank:]
        loc = loc[:, :self.rank], loc[:, self.rank:]
        right = right[:, :self.rank], right[:, self.rank:]

        user1 = self.embeddings[10](similar[:, 0])
        user1 = nn.functional.normalize(user1,dim=1)
        sim = self.embeddings[11](similar[:, 1])
        right_sim = self.embeddings[10].weight
        user1 = user1[:, :self.rank], user1[:, self.rank:]
        sim = sim[:, :self.rank], sim[:, self.rank:]
        right_sim = right_sim[:, :self.rank], right_sim[:, self.rank:]

        sim_score = (user1[0] * sim[0] - user1[1] * sim[1]) @ right_sim[0].t() + (user1[1] * sim[0] + user1[0] * sim[1]) @ right_sim[1].t()


        return ((origin[0] * tran[0] - origin[1] * tran[1]) @ right[0].t() + (origin[1] * tran[0] + origin[0] * tran[1]) @ right[1].t() 
        ),sim_score, (
                   torch.sqrt(origin[0] ** 2 + origin[1] ** 2),
                   torch.sqrt(tran[0] ** 2 + tran[1] ** 2),
                   torch.sqrt(loc[0] ** 2 + loc[1] ** 2),
               )

    def forward1(self, x):
        origin = self.embeddings[0](x[:, 6])
        loc = self.embeddings[2](x[:, 3])
        inv = self.embeddings[3](x[:, 5])
        acc = self.embeddings[4](x[:, 2])
        delta = self.diachronic(x[:, 7])
        tim1 = self.embeddings[8](x[:, 4])
        tim = self.embeddings[9](x[:, 1])
        
        right_ots = self.embeddings[8].weight
        right_ts = self.embeddings[9].weight
        right_loc = self.embeddings[2].weight

        origin = origin[:, :self.rank], origin[:, self.rank:]
        loc = loc[:, :self.rank], loc[:, self.rank:]
        delta = delta[:, :self.rank], delta[:, self.rank:]
        invd = inv[:, :int(self.user_ratio * self.rank)], inv[:, self.rank:self.rank + int(self.user_ratio * self.rank)]
        invs = inv[:, int(self.user_ratio * self.rank):self.rank], inv[:, self.rank + int(self.user_ratio * self.rank):]
        inv = tuple(torch.cat((x, y), 1) for x, y in zip(invd, invs))
        accd = acc[:, :int(self.user_ratio * self.rank)], acc[:, self.rank:self.rank + int(self.user_ratio * self.rank)]
        accs = acc[:, int(self.user_ratio * self.rank):self.rank], acc[:, self.rank + int(self.user_ratio * self.rank):]
        acc = tuple(torch.cat((x, y), 1) for x, y in zip(accd, accs))

        tim1 = tim1[:, :self.rank], tim1[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        right_ots = right_ots[:, :self.rank], right_ots[:, self.rank:]
        right_ts = right_ts[:, :self.rank], right_ts[:, self.rank:]
        right_loc = right_loc[:, :self.rank], right_loc[:, self.rank:]

        return ((origin[0] * acc[0] - origin[1] * acc[1]) @ right_ots[0].t() + (origin[1] * acc[0] + origin[0] * acc[1]) @ right_ots[1].t() ),(
            (tim1[0] * delta[0] - tim1[1] * delta[1]) @ right_ts[0].t() + (tim1[1] * delta[0] + tim1[0] * delta[1]) @ right_ts[1].t()),(
              (tim[0] * inv[0] - tim[1] * inv[1]) @ right_loc[0].t() + (tim[1] * inv[0] + tim[0] * inv[1]) @ right_loc[1].t() )  

    def get_queries(self, queries: torch.Tensor):
        origin = self.embeddings[0](queries[:, 6])
        tran = self.embeddings[1](queries[:, 8])
        trand = tran[:, :int(self.user_ratio * self.rank)], tran[:, self.rank:self.rank + int(self.user_ratio * self.rank)]
        trans = tran[:, int(self.user_ratio * self.rank):self.rank], tran[:, self.rank + int(self.user_ratio * self.rank):]
        tran = tuple(torch.cat((x, y), 1) for x, y in zip(trand, trans))
  
        origin = origin[:, :self.rank], origin[:, self.rank:]
        locs = self.embeddings[2].weight
        locs = locs[:, :self.rank], locs[:, self.rank:]

        return  ((origin[0] * tran[0] - origin[1] * tran[1]) @ locs[0].t() + (origin[1] * tran[0] + origin[0] * tran[1]) @ locs[1].t() )

class MComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int, int, int, int, int], rank: int,
             test_transition=True, init_size: float = 1e-3
    ):
        
        super(MComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank
       
        self.time_nl = torch.sin
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank)
            for s in [sizes[2], sizes[6], sizes[2], sizes[5], sizes[3], sizes[4], sizes[4], sizes[4], sizes[1], sizes[1],sizes[0],sizes[7]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size
        self.embeddings[4].weight.data *= init_size
        self.embeddings[5].weight.data *= init_size
        self.embeddings[6].weight.data *= init_size
        self.embeddings[7].weight.data *= init_size
        self.embeddings[8].weight.data *= init_size
        self.embeddings[9].weight.data *= init_size
        self.embeddings[10].weight.data *= init_size
        self.embeddings[11].weight.data *= init_size
        self.bn = nn.BatchNorm1d(sizes[2])
        self.test_transition = test_transition

    @staticmethod
    def has_time():
        return True

    def proj_user(self, ent, user):
        ent = ent - user*torch.sum(user*ent,dim=1, keepdim=True)
        return ent

    def proj_users(self, pois, user):
        tmp = torch.einsum('ik,kj->ij',user,torch.transpose(pois,1,0))
        pois = pois - torch.einsum('bij,bjk->bik',tmp[:,:,np.newaxis],user[:,np.newaxis,:])
        return pois

    def diachronic(self, x):
        delta_a = self.embeddings[5](x)
        delta_freq = self.embeddings[6](x)
        delta_phi = self.embeddings[7](x)
        delta = delta_a * self.time_nl(delta_freq * x.type(torch.FloatTensor).cuda().reshape((-1,1))  + delta_phi)
        return delta

    def forward(self, x, similar):
        origin = self.embeddings[0](x[:, 6])
        tran = self.embeddings[1](x[:, 8])
        loc = self.embeddings[2](x[:, 3])
        tim = self.embeddings[9](x[:, 1])
        inv = self.embeddings[3](x[:, 5])
        user = self.embeddings[10](x[:, 0])
        user = nn.functional.normalize(user,dim=1)
        
        origin = self.proj_user(origin,user)
        tran = self.proj_user(tran,user)
        loc = self.proj_user(loc,user)
        tim = self.proj_user(tim,user)
        inv = self.proj_user(inv,user)
        right = self.proj_users(self.embeddings[2].weight,user)

        origin = origin[:, :self.rank], origin[:, self.rank:]
        tran = tran[:, :self.rank], tran[:, self.rank:]
        loc = loc[:, :self.rank], loc[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        inv = inv[:, :self.rank], inv[:, self.rank:]
        right = right[:, :, :self.rank], right[:, :, self.rank:]
        
        user1 = self.embeddings[10](similar[:, 0])
        user1 = nn.functional.normalize(user1,dim=1)
        sim = self.embeddings[11](similar[:, 1])
        right_sim = self.embeddings[10].weight
        user1 = user1[:, :self.rank], user1[:, self.rank:]
        sim = sim[:, :self.rank], sim[:, self.rank:]
        right_sim = right_sim[:, :self.rank], right_sim[:, self.rank:]

        sim_score = (user1[0] * sim[0] - user1[1] * sim[1]) @ right_sim[0].t() + (user1[1] * sim[0] + user1[0] * sim[1]) @ right_sim[1].t()


        if not self.test_transition:
            score = (torch.squeeze(torch.bmm((tim[0] * inv[0] - tim[1] * inv[1]).view(-1,1,self.rank), torch.transpose(right[0],1,2))) +
             torch.squeeze(torch.bmm((tim[0] * inv[1] + tim[1] * inv[0]).view(-1,1,self.rank), torch.transpose(right[1],1,2))))
        else:
            score = (torch.squeeze(torch.bmm((origin[0] * tran[0] - origin[1] * tran[1]).view(-1,1,self.rank), torch.transpose(right[0],1,2))) +
             torch.squeeze(torch.bmm((origin[0] * tran[1] + origin[1] * tran[0]).view(-1,1,self.rank), torch.transpose(right[1],1,2))))

        return score, sim_score,(
                   torch.sqrt(origin[0] ** 2 + origin[1] ** 2),
                   torch.sqrt(tran[0] ** 2 + tran[1] ** 2),
                   torch.sqrt(loc[0] ** 2 + loc[1] ** 2),
               )

    def forward1(self, x):
        origin = self.embeddings[0](x[:, 6])
        loc = self.embeddings[2](x[:, 3])
        inv = self.embeddings[3](x[:, 5])
        acc = self.embeddings[4](x[:, 2])
        delta = self.diachronic(x[:, 7])
        tim1 = self.embeddings[8](x[:, 4])
        tim = self.embeddings[9](x[:, 1])
        user = self.embeddings[10](x[:, 0])
        user = nn.functional.normalize(user,dim=1)
        
        origin = self.proj_user(origin,user)
        loc = self.proj_user(loc,user)
        inv = self.proj_user(inv,user)
        acc = self.proj_user(acc,user)
        delta = self.proj_user(delta,user)
        tim1 = self.proj_user(tim1,user)
        tim = self.proj_user(tim,user)
        
        right_ots = self.proj_users(self.embeddings[8].weight,user)
        right_ts = self.proj_users(self.embeddings[9].weight,user)
        right_loc = self.proj_users(self.embeddings[2].weight,user)

        origin = origin[:, :self.rank], origin[:, self.rank:]
        loc = loc[:, :self.rank], loc[:, self.rank:]
        inv = inv[:, :self.rank], inv[:, self.rank:]
        acc = acc[:, :self.rank], acc[:, self.rank:]
        delta = delta[:, :self.rank], delta[:, self.rank:]
        tim1 = tim1[:, :self.rank], tim1[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        right_ots = right_ots[:, :, :self.rank], right_ots[:, :, self.rank:]
        right_ts = right_ts[:, :, :self.rank], right_ts[:, :, self.rank:]
        right_loc = right_loc[:, :, :self.rank], right_loc[:, :, self.rank:]

        return (torch.squeeze(torch.bmm((origin[0] * acc[0] - origin[1] * acc[1]).view(-1,1,self.rank), torch.transpose(right_ots[0],1,2))) +
             torch.squeeze(torch.bmm((origin[0] * acc[1] + origin[1] * acc[0]).view(-1,1,self.rank), torch.transpose(right_ots[1],1,2)))), (
             torch.squeeze(torch.bmm((tim1[0] * delta[0] - tim1[1] * delta[1]).view(-1,1,self.rank), torch.transpose(right_ts[0],1,2))) +
             torch.squeeze(torch.bmm((tim1[0] * delta[1] + tim1[1] * delta[0]).view(-1,1,self.rank), torch.transpose(right_ts[1],1,2))) ),(
             torch.squeeze(torch.bmm((tim[0] * inv[0] - tim[1] * inv[1]).view(-1,1,self.rank), torch.transpose(right_loc[0],1,2))) +
             torch.squeeze(torch.bmm((tim[0] * inv[1] + tim[1] * inv[0]).view(-1,1,self.rank), torch.transpose(right_loc[1],1,2)))
             )  

    def get_queries(self, queries: torch.Tensor):
        origin = self.embeddings[0](queries[:, 6])
        tran = self.embeddings[1](queries[:, 8])
        tim = self.embeddings[9](queries[:, 1])
        inv = self.embeddings[3](queries[:, 5])
        user = self.embeddings[10](queries[:, 0])
        user = nn.functional.normalize(user,dim=1)
        
        origin = self.proj_user(origin,user)
        tran = self.proj_user(tran,user)
        tim = self.proj_user(tim,user)
        inv = self.proj_user(inv,user)
        locs = self.proj_users(self.embeddings[2].weight,user)

        origin = origin[:, :self.rank], origin[:, self.rank:]
        tran = tran[:, :self.rank], tran[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        inv = inv[:, :self.rank], inv[:, self.rank:]
        locs = locs[:, :, :self.rank], locs[:, :, self.rank:]

        if not self.test_transition:
            score = (torch.squeeze(torch.bmm((tim[0] * inv[0] - tim[1] * inv[1]).view(-1,1,self.rank), torch.transpose(locs[0],1,2))) +
             torch.squeeze(torch.bmm((tim[0] * inv[1] + tim[1] * inv[0]).view(-1,1,self.rank), torch.transpose(locs[1],1,2))))
        else:
            score = (torch.squeeze(torch.bmm((origin[0] * tran[0] - origin[1] * tran[1]).view(-1,1,self.rank), torch.transpose(locs[0],1,2))) +
             torch.squeeze(torch.bmm((origin[0] * tran[1] + origin[1] * tran[0]).view(-1,1,self.rank), torch.transpose(locs[1],1,2))))

        return  score

class MComplEx11(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int, int, int, int, int], rank: int,
              test_transition = True,init_size: float = 1e-3
    ):
        
        super(MComplEx11, self).__init__()
        self.sizes = sizes
        self.rank = rank
       
        self.time_nl = torch.sin
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank)
            for s in [sizes[2], sizes[6], sizes[2], sizes[5], sizes[3], sizes[4], sizes[4], sizes[4], sizes[1], sizes[1],sizes[0],sizes[7]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size
        self.embeddings[4].weight.data *= init_size
        self.embeddings[5].weight.data *= init_size
        self.embeddings[6].weight.data *= init_size
        self.embeddings[7].weight.data *= init_size
        self.embeddings[8].weight.data *= init_size
        self.embeddings[9].weight.data *= init_size
        self.embeddings[10].weight.data *= init_size
        self.embeddings[11].weight.data *= init_size
        self.bn = nn.BatchNorm1d(sizes[2])
        self.hidden1 = torch.nn.Linear(5*rank, 200)
        # self.hidden2 = torch.nn.Linear(120, 200)
        self.out = torch.nn.Linear(200, rank)
        self.test_transition = test_transition

    @staticmethod
    def has_time():
        return True

    def diachronic(self, x):
        delta_a = self.embeddings[5](x)
        delta_freq = self.embeddings[6](x)
        delta_phi = self.embeddings[7](x)
        delta = delta_a * self.time_nl(delta_freq * x.type(torch.FloatTensor).cuda().reshape((-1,1))  + delta_phi)
        return delta

    def proj_user(self, ent, user):
        ent = ent - user*torch.sum(user*ent,dim=1, keepdim=True)
        return ent

    def proj_users(self, pois, user):
        tmp = torch.einsum('ik,kj->ij',user,torch.transpose(pois,1,0))
        pois = pois - torch.einsum('bij,bjk->bik',tmp[:,:,np.newaxis],user[:,np.newaxis,:])
        return pois

    def forward(self, x, similar):
        origin = self.embeddings[0](x[:, 6])
        tran = self.embeddings[1](x[:, 8])
        loc = self.embeddings[2](x[:, 3])
        inv = self.embeddings[3](x[:, 5])
        acc = self.embeddings[4](x[:, 2])
        delta = self.diachronic(x[:, 7])
        tim1 = self.embeddings[8](x[:, 4])
        tim = self.embeddings[9](x[:, 1])
        user = self.embeddings[10](x[:, 0])
        user = nn.functional.normalize(user,dim=1)
        
        origin = self.proj_user(origin,user)
        tran = self.proj_user(tran,user)
        loc = self.proj_user(loc,user)
        inv = self.proj_user(inv,user)
        acc = self.proj_user(acc,user)
        delta = self.proj_user(delta,user)
        tim1 = self.proj_user(tim1,user)
        tim = self.proj_user(tim,user)
        right = self.proj_users(self.embeddings[2].weight,user)

        origin = origin[:, :self.rank], origin[:, self.rank:]
        tran = tran[:, :self.rank], tran[:, self.rank:]
        loc = loc[:, :self.rank], loc[:, self.rank:]
        inv = inv[:, :self.rank], inv[:, self.rank:]
        acc = acc[:, :self.rank], acc[:, self.rank:]
        delta = delta[:, :self.rank], delta[:, self.rank:]
        tim1 = tim1[:, :self.rank], tim1[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        right = right[:, :, :self.rank], right[:, :, self.rank:]

        # path1_rel = acc[0] * delta[0] * inv[0], acc[1] * delta[1] * inv[1]
        
        path1 = torch.cat([acc[0],tim1[0],delta[0],tim[0],inv[0]],1),torch.cat([acc[1],tim1[1],delta[1],tim[1],inv[1]],1)
        rel_path1 = torch.cat([self.out(F.relu(self.hidden1(path1[0]))),self.out(F.relu(self.hidden1(path1[1])))],1)
        rel_path1 = self.out(F.relu(self.hidden1(path1[0]))),self.out(F.relu(self.hidden1(path1[1])))
        eng1 = rel_path1[0] - tran[0], rel_path1[1] - tran[1]
        
        user1 = self.embeddings[10](similar[:, 0])
        user1 = nn.functional.normalize(user1,dim=1)
        sim = self.embeddings[11](similar[:, 1])
        right_sim = self.embeddings[10].weight
        user1 = user1[:, :self.rank], user1[:, self.rank:]
        sim = sim[:, :self.rank], sim[:, self.rank:]
        right_sim = right_sim[:, :self.rank], right_sim[:, self.rank:]

        sim_score = (user1[0] * sim[0] - user1[1] * sim[1]) @ right_sim[0].t() + (user1[1] * sim[0] + user1[0] * sim[1]) @ right_sim[1].t()



        if not self.test_transition:
            score = (torch.squeeze(torch.bmm((tim[0] * inv[0] - tim[1] * inv[1]).view(-1,1,self.rank), torch.transpose(right[0],1,2))) +
             torch.squeeze(torch.bmm((tim[0] * inv[1] + tim[1] * inv[0]).view(-1,1,self.rank), torch.transpose(right[1],1,2))))
        else:
            score = (torch.squeeze(torch.bmm((origin[0] * tran[0] - origin[1] * tran[1]).view(-1,1,self.rank), torch.transpose(right[0],1,2))) +
             torch.squeeze(torch.bmm((origin[0] * tran[1] + origin[1] * tran[0]).view(-1,1,self.rank), torch.transpose(right[1],1,2))))

        return score,sim_score, (
             torch.squeeze(torch.bmm((origin[0] * rel_path1[0] - origin[1] * rel_path1[1]).view(-1,1,self.rank), torch.transpose(right[0],1,2))) +
             torch.squeeze(torch.bmm((origin[0] * rel_path1[1] + origin[1] * rel_path1[0]).view(-1,1,self.rank), torch.transpose(right[1],1,2))) ),(
                
                   torch.sqrt(origin[0] ** 2 + origin[1] ** 2),
                   torch.sqrt(tran[0] ** 2 + tran[1] ** 2),
                   torch.sqrt(acc[0] ** 2 + acc[1] ** 2),
                   torch.sqrt(inv[0] ** 2 + inv[1] ** 2),
                   torch.sqrt(loc[0] ** 2 + loc[1] ** 2),
                   torch.sqrt(delta[0] ** 2 + delta[1] ** 2),
                   torch.sqrt(tim1[0] ** 2 + tim1[1] ** 2),
                   torch.sqrt(tim[0] ** 2 + tim[1] ** 2),

               ), (torch.sum(torch.sqrt(eng1[0] ** 2 + eng1[1] ** 2)) / len(eng1) )

    def forward1(self, x):
        origin = self.embeddings[0](x[:, 6])
        loc = self.embeddings[2](x[:, 3])
        inv = self.embeddings[3](x[:, 5])
        acc = self.embeddings[4](x[:, 2])
        delta = self.diachronic(x[:, 7])
        tim1 = self.embeddings[8](x[:, 4])
        tim = self.embeddings[9](x[:, 1])
        user = self.embeddings[10](x[:, 0])
        user = nn.functional.normalize(user,dim=1)
        
        origin = self.proj_user(origin,user)
        loc = self.proj_user(loc,user)
        inv = self.proj_user(inv,user)
        acc = self.proj_user(acc,user)
        delta = self.proj_user(delta,user)
        tim1 = self.proj_user(tim1,user)
        tim = self.proj_user(tim,user)
        
        right_ots = self.proj_users(self.embeddings[8].weight,user)
        right_ts = self.proj_users(self.embeddings[9].weight,user)
        right_loc = self.proj_users(self.embeddings[2].weight,user)

        origin = origin[:, :self.rank], origin[:, self.rank:]
        loc = loc[:, :self.rank], loc[:, self.rank:]
        inv = inv[:, :self.rank], inv[:, self.rank:]
        acc = acc[:, :self.rank], acc[:, self.rank:]
        delta = delta[:, :self.rank], delta[:, self.rank:]
        tim1 = tim1[:, :self.rank], tim1[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        right_ots = right_ots[:, :, :self.rank], right_ots[:, :, self.rank:]
        right_ts = right_ts[:, :, :self.rank], right_ts[:, :, self.rank:]
        right_loc = right_loc[:, :, :self.rank], right_loc[:, :, self.rank:]

        return (torch.squeeze(torch.bmm((origin[0] * acc[0] - origin[1] * acc[1]).view(-1,1,self.rank), torch.transpose(right_ots[0],1,2))) +
             torch.squeeze(torch.bmm((origin[0] * acc[1] + origin[1] * acc[0]).view(-1,1,self.rank), torch.transpose(right_ots[1],1,2)))), (
             torch.squeeze(torch.bmm((tim1[0] * delta[0] - tim1[1] * delta[1]).view(-1,1,self.rank), torch.transpose(right_ts[0],1,2))) +
             torch.squeeze(torch.bmm((tim1[0] * delta[1] + tim1[1] * delta[0]).view(-1,1,self.rank), torch.transpose(right_ts[1],1,2))) ),(
             torch.squeeze(torch.bmm((tim[0] * inv[0] - tim[1] * inv[1]).view(-1,1,self.rank), torch.transpose(right_loc[0],1,2))) +
             torch.squeeze(torch.bmm((tim[0] * inv[1] + tim[1] * inv[0]).view(-1,1,self.rank), torch.transpose(right_loc[1],1,2)))
             )     

    def get_queries(self, queries: torch.Tensor):
        origin = self.embeddings[0](queries[:, 6])
        tran = self.embeddings[1](queries[:, 8])
        inv = self.embeddings[3](queries[:, 5])
        acc = self.embeddings[4](queries[:, 2])
        delta = self.diachronic(queries[:, 7])
        tim1 = self.embeddings[8](queries[:, 4])
        tim = self.embeddings[9](queries[:, 1])
        user = self.embeddings[10](queries[:, 0])
        user = nn.functional.normalize(user,dim=1)

        origin = self.proj_user(origin,user)
        tran = self.proj_user(tran,user)
        inv = self.proj_user(inv,user)
        acc = self.proj_user(acc,user)
        delta = self.proj_user(delta,user)
        tim1 = self.proj_user(tim1,user)
        tim = self.proj_user(tim,user)
        locs = self.proj_users(self.embeddings[2].weight,user)

        origin = origin[:, :self.rank], origin[:, self.rank:]
        tran = tran[:, :self.rank], tran[:, self.rank:]
        inv = inv[:, :self.rank], inv[:, self.rank:]
        acc = acc[:, :self.rank], acc[:, self.rank:]
        delta = delta[:, :self.rank], delta[:, self.rank:]
        tim1 = tim1[:, :self.rank], tim1[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        locs = locs[:, :, :self.rank], locs[:, :, self.rank:]
        path1 = torch.cat([acc[0],tim1[0],delta[0],tim[0],inv[0]],1),torch.cat([acc[1],tim1[1],delta[1],tim[1],inv[1]],1)
        rel_path1 = torch.cat([self.out(F.relu(self.hidden1(path1[0]))),self.out(F.relu(self.hidden1(path1[1])))],1)
        rel_path1 = self.out(F.relu(self.hidden1(path1[0]))),self.out(F.relu(self.hidden1(path1[1])))
        eng1 = rel_path1[0] - tran[0], rel_path1[1] - tran[1]


        if not self.test_transition:
            score = (torch.squeeze(torch.bmm((tim[0] * inv[0] - tim[1] * inv[1]).view(-1,1,self.rank), torch.transpose(locs[0],1,2))) +
             torch.squeeze(torch.bmm((tim[0] * inv[1] + tim[1] * inv[0]).view(-1,1,self.rank), torch.transpose(locs[1],1,2))))
        else:
            score = (torch.squeeze(torch.bmm((origin[0] * tran[0] - origin[1] * tran[1]).view(-1,1,self.rank), torch.transpose(locs[0],1,2))) +
             torch.squeeze(torch.bmm((origin[0] * tran[1] + origin[1] * tran[0]).view(-1,1,self.rank), torch.transpose(locs[1],1,2))))

        return  score

class MComplEx12(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int, int, int, int, int], rank: int,
             test_transition = True, init_size: float = 1e-3
    ):

        super(MComplEx12, self).__init__()
        self.sizes = sizes
        self.rank = rank
       
        self.time_nl = torch.sin
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2*rank, sparse=True)
            for s in [sizes[2], sizes[6], sizes[2], sizes[5], sizes[3], sizes[4], sizes[4], sizes[4], sizes[1], sizes[1],sizes[0],sizes[7]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size
        self.embeddings[4].weight.data *= init_size
        self.embeddings[5].weight.data *= init_size
        self.embeddings[6].weight.data *= init_size
        self.embeddings[7].weight.data *= init_size
        self.embeddings[8].weight.data *= init_size
        self.embeddings[9].weight.data *= init_size
        self.embeddings[10].weight.data *= init_size
        self.embeddings[11].weight.data *= init_size
        self.bn = nn.BatchNorm1d(sizes[2])
        self.bn = nn.BatchNorm1d(sizes[2])
        self.hidden1 = torch.nn.Linear(rank*5, 200)
        self.hidden2 = torch.nn.Linear(rank *3, 200)
        self.out = torch.nn.Linear(200, rank)
        self.test_transition =test_transition

       
    @staticmethod
    def has_time():
        return True

    def diachronic(self, x):
        delta_a = self.embeddings[5](x)
        delta_freq = self.embeddings[6](x)
        delta_phi = self.embeddings[7](x)
        delta = delta_a * self.time_nl(delta_freq * x.type(torch.FloatTensor).cuda().reshape((-1,1))  + delta_phi)
        return delta

    def proj_user(self, ent, user):
        ent = ent - user*torch.sum(user*ent,dim=1, keepdim=True)
        return ent

    def proj_users(self, pois, user):
        tmp = torch.mm(user,pois.transpose(0,1))
        pois = pois - torch.bmm(tmp[:,:,np.newaxis],user[:,np.newaxis,:])
        return pois

    
    def forward(self, x, path2_origin, path2_mid, path2_user, similar):
        origin = self.embeddings[0](x[:, 6])
        tran = self.embeddings[1](x[:, 8])
        loc = self.embeddings[2](x[:, 3])
        inv = self.embeddings[3](x[:, 5])
        acc = self.embeddings[4](x[:, 2])
        delta = self.diachronic(x[:, 7])
        tim1 = self.embeddings[8](x[:, 4])
        tim = self.embeddings[9](x[:, 1])
        user = self.embeddings[10](x[:, 0])
        
        origin = self.proj_user(origin,user)
        tran = self.proj_user(tran,user)
        loc = self.proj_user(loc,user)
        inv = self.proj_user(inv,user)
        acc = self.proj_user(acc,user)
        delta = self.proj_user(delta,user)
        tim1 = self.proj_user(tim1,user)
        tim = self.proj_user(tim,user)

        right = self.proj_users(self.embeddings[2].weight,user)

        origin = origin[:, :self.rank], origin[:, self.rank:]
        tran = tran[:, :self.rank], tran[:, self.rank:]
        loc = loc[:, :self.rank], loc[:, self.rank:]
        inv = inv[:, :self.rank], inv[:, self.rank:]
        acc = acc[:, :self.rank], acc[:, self.rank:]
        delta = delta[:, :self.rank], delta[:, self.rank:]
        tim1 = tim1[:, :self.rank], tim1[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        right = right[:, :, :self.rank], right[:, :, self.rank:]

        user2 = self.embeddings[10](path2_user)
        path2_origin = self.proj_user(self.embeddings[0](path2_origin),user2)
        path2_mid = self.proj_user(self.embeddings[0](path2_mid),user2)
        path2_tran = self.proj_user(self.embeddings[1](x[:,8][0].repeat(len(path2_origin))),user2)
        path2_right = self.proj_users(self.embeddings[2].weight,user2)
        path2_origin = path2_origin[:, :self.rank], path2_origin[:, self.rank:]
        path2_mid = path2_mid[:, :self.rank], path2_mid[:, self.rank:]
        path2_tran = path2_tran[:, :self.rank], path2_tran[:, self.rank:]
        path2_right = path2_right[:, :,:self.rank], path2_right[:,:, self.rank:]

        path2 = torch.cat([path2_tran[0],path2_tran[0],path2_mid[0]],1),torch.cat([path2_tran[1],path2_tran[1],path2_mid[1]],1)
        rel_path2 = self.out(F.relu(self.hidden2(path2[0]))),self.out(F.relu(self.hidden2(path2[1])))
        eng2 = rel_path2[0] - path2_tran[0], rel_path2[1] - path2_tran[1]

        if not self.test_transition:
            score = (torch.squeeze(torch.bmm((tim[0] * inv[0] - tim[1] * inv[1]).view(-1,1,self.rank), torch.transpose(right[0],1,2))) +
             torch.squeeze(torch.bmm((tim[0] * inv[1] + tim[1] * inv[0]).view(-1,1,self.rank), torch.transpose(right[1],1,2))))
        else:
            score = (torch.squeeze(torch.bmm((origin[0] * tran[0] - origin[1] * tran[1]).view(-1,1,self.rank), torch.transpose(right[0],1,2))) +
             torch.squeeze(torch.bmm((origin[0] * tran[1] + origin[1] * tran[0]).view(-1,1,self.rank), torch.transpose(right[1],1,2))))
        
        user1 = self.embeddings[10](similar[:, 0])
        user1 = nn.functional.normalize(user1,dim=1)
        sim = self.embeddings[11](similar[:, 1])
        right_sim = self.embeddings[10].weight
        user1 = user1[:, :self.rank], user1[:, self.rank:]
        sim = sim[:, :self.rank], sim[:, self.rank:]
        right_sim = right_sim[:, :self.rank], right_sim[:, self.rank:]

        sim_score = (user1[0] * sim[0] - user1[1] * sim[1]) @ right_sim[0].t() + (user1[1] * sim[0] + user1[0] * sim[1]) @ right_sim[1].t()

        return score,sim_score, (
              torch.squeeze(torch.bmm((path2_origin[0] * rel_path2[0] - path2_origin[1] * rel_path2[1]).view(-1,1,self.rank), torch.transpose(path2_right[0],1,2))) +
             torch.squeeze(torch.bmm((path2_origin[0] * rel_path2[1] + path2_origin[1] * rel_path2[0]).view(-1,1,self.rank), torch.transpose(path2_right[1],1,2))) ),(
                     
                   torch.sqrt(origin[0] ** 2 + origin[1] ** 2),
                   torch.sqrt(tran[0] ** 2 + tran[1] ** 2),
                   torch.sqrt(acc[0] ** 2 + acc[1] ** 2),
                   torch.sqrt(inv[0] ** 2 + inv[1] ** 2),
                   torch.sqrt(loc[0] ** 2 + loc[1] ** 2),
                   torch.sqrt(delta[0] ** 2 + delta[1] ** 2),
                   torch.sqrt(tim1[0] ** 2 + tim1[1] ** 2),
                   torch.sqrt(tim[0] ** 2 + tim[1] ** 2),

               ), ((torch.sum(torch.sqrt(eng2[0] ** 2 + eng2[1] ** 2)) / len(eng2) ))


    def forward1(self, x):
        origin = self.embeddings[0](x[:, 6])
        loc = self.embeddings[2](x[:, 3])
        inv = self.embeddings[3](x[:, 5])
        acc = self.embeddings[4](x[:, 2])
        delta = self.diachronic(x[:, 7])
        tim1 = self.embeddings[8](x[:, 4])
        tim = self.embeddings[9](x[:, 1])
        user = self.embeddings[10](x[:, 0])
        user = nn.functional.normalize(user,dim=1)
        
        origin = self.proj_user(origin,user)
        loc = self.proj_user(loc,user)
        inv = self.proj_user(inv,user)
        acc = self.proj_user(acc,user)
        delta = self.proj_user(delta,user)
        tim1 = self.proj_user(tim1,user)
        tim = self.proj_user(tim,user)
        
        right_ots = self.proj_users(self.embeddings[8].weight,user)
        right_ts = self.proj_users(self.embeddings[9].weight,user)
        right_loc = self.proj_users(self.embeddings[2].weight,user)

        origin = origin[:, :self.rank], origin[:, self.rank:]
        loc = loc[:, :self.rank], loc[:, self.rank:]
        inv = inv[:, :self.rank], inv[:, self.rank:]
        acc = acc[:, :self.rank], acc[:, self.rank:]
        delta = delta[:, :self.rank], delta[:, self.rank:]
        tim1 = tim1[:, :self.rank], tim1[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        right_ots = right_ots[:, :, :self.rank], right_ots[:, :, self.rank:]
        right_ts = right_ts[:, :, :self.rank], right_ts[:, :, self.rank:]
        right_loc = right_loc[:, :, :self.rank], right_loc[:, :, self.rank:]

        return (torch.squeeze(torch.bmm((origin[0] * acc[0] - origin[1] * acc[1]).view(-1,1,self.rank), torch.transpose(right_ots[0],1,2))) +
             torch.squeeze(torch.bmm((origin[0] * acc[1] + origin[1] * acc[0]).view(-1,1,self.rank), torch.transpose(right_ots[1],1,2)))), (
             torch.squeeze(torch.bmm((tim1[0] * delta[0] - tim1[1] * delta[1]).view(-1,1,self.rank), torch.transpose(right_ts[0],1,2))) +
             torch.squeeze(torch.bmm((tim1[0] * delta[1] + tim1[1] * delta[0]).view(-1,1,self.rank), torch.transpose(right_ts[1],1,2))) ),(
             torch.squeeze(torch.bmm((tim[0] * inv[0] - tim[1] * inv[1]).view(-1,1,self.rank), torch.transpose(right_loc[0],1,2))) +
             torch.squeeze(torch.bmm((tim[0] * inv[1] + tim[1] * inv[0]).view(-1,1,self.rank), torch.transpose(right_loc[1],1,2)))
             )     
   

    def get_queries(self, queries: torch.Tensor):
        origin = self.embeddings[0](queries[:, 6])
        tran = self.embeddings[1](queries[:, 8])
        inv = self.embeddings[3](queries[:, 5])
        acc = self.embeddings[4](queries[:, 2])
        delta = self.diachronic(queries[:, 7])
        tim1 = self.embeddings[8](queries[:, 4])
        tim = self.embeddings[9](queries[:, 1])
        user = self.embeddings[10](queries[:, 0])

        origin = self.proj_user(origin,user)
        tran = self.proj_user(tran,user)
        inv = self.proj_user(inv,user)
        acc = self.proj_user(acc,user)
        delta = self.proj_user(delta,user)
        tim1 = self.proj_user(tim1,user)
        tim = self.proj_user(tim,user)
        locs = self.proj_users(self.embeddings[2].weight,user)

        origin = origin[:, :self.rank], origin[:, self.rank:]
        tran = tran[:, :self.rank], tran[:, self.rank:]

        inv = inv[:, :self.rank], inv[:, self.rank:]
        acc = acc[:, :self.rank], acc[:, self.rank:]
        delta = delta[:, :self.rank], delta[:, self.rank:]
        tim1 = tim1[:, :self.rank], tim1[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        locs = locs[:, :, :self.rank], locs[:, :, self.rank:]

        if not self.test_transition:
            score = (torch.squeeze(torch.bmm((tim[0] * inv[0] - tim[1] * inv[1]).view(-1,1,self.rank), torch.transpose(locs[0],1,2))) +
             torch.squeeze(torch.bmm((tim[0] * inv[1] + tim[1] * inv[0]).view(-1,1,self.rank), torch.transpose(locs[1],1,2))))
        else:
            score = (torch.squeeze(torch.bmm((origin[0] * tran[0] - origin[1] * tran[1]).view(-1,1,self.rank), torch.transpose(locs[0],1,2))) +
             torch.squeeze(torch.bmm((origin[0] * tran[1] + origin[1] * tran[0]).view(-1,1,self.rank), torch.transpose(locs[1],1,2))))


        return score


class MComplEx21(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int, int, int, int, int], rank: int,
            test_transition = True, init_size: float = 1e-3
    ):

        super(MComplEx21, self).__init__()
        self.sizes = sizes
        self.rank = rank
       
        self.time_nl = torch.sin
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2*rank, sparse=True)
            for s in [sizes[2], sizes[6], sizes[2], sizes[5], sizes[3], sizes[4], sizes[4], sizes[4], sizes[1], sizes[1],sizes[0],sizes[7]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size
        self.embeddings[4].weight.data *= init_size
        self.embeddings[5].weight.data *= init_size
        self.embeddings[6].weight.data *= init_size
        self.embeddings[7].weight.data *= init_size
        self.embeddings[8].weight.data *= init_size
        self.embeddings[9].weight.data *= init_size
        self.embeddings[10].weight.data *= init_size
        self.embeddings[11].weight.data *= init_size
        self.bn = nn.BatchNorm1d(sizes[2])
        self.bn = nn.BatchNorm1d(sizes[2])
        self.hidden1 = torch.nn.Linear(3*rank, 200)
        self.hidden2 = torch.nn.Linear(2*rank, 200)
        self.out = torch.nn.Linear(200, rank)
        self.test_transition = test_transition

       
    @staticmethod
    def has_time():
        return True

    def diachronic(self, x):
        delta_a = self.embeddings[5](x)
        delta_freq = self.embeddings[6](x)
        delta_phi = self.embeddings[7](x)
        delta = delta_a * self.time_nl(delta_freq * x.type(torch.FloatTensor).cuda().reshape((-1,1))  + delta_phi)
        return delta

    def proj_user(self, ent, user):
        ent = ent - user*torch.sum(user*ent,dim=1, keepdim=True)
        return ent

    def proj_users(self, pois, user):
        tmp = torch.mm(user,pois.transpose(0,1))
        pois = pois - torch.bmm(tmp[:,:,np.newaxis],user[:,np.newaxis,:])
        return pois

    
    def forward(self, x, path2_origin, path2_mid, path2_user, similar):
        origin = self.embeddings[0](x[:, 6])
        tran = self.embeddings[1](x[:, 8])
        loc = self.embeddings[2](x[:, 3])
        inv = self.embeddings[3](x[:, 5])
        acc = self.embeddings[4](x[:, 2])
        delta = self.diachronic(x[:, 7])
        tim1 = self.embeddings[8](x[:, 4])
        tim = self.embeddings[9](x[:, 1])
        user = self.embeddings[10](x[:, 0])
        
        origin = self.proj_user(origin,user)
        tran = self.proj_user(tran,user)
        loc = self.proj_user(loc,user)
        inv = self.proj_user(inv,user)
        acc = self.proj_user(acc,user)
        delta = self.proj_user(delta,user)
        tim1 = self.proj_user(tim1,user)
        tim = self.proj_user(tim,user)

        right = self.proj_users(self.embeddings[2].weight,user)


        origin = origin[:, :self.rank], origin[:, self.rank:]
        tran = tran[:, :self.rank], tran[:, self.rank:]
        loc = loc[:, :self.rank], loc[:, self.rank:]
        inv = inv[:, :self.rank], inv[:, self.rank:]
        acc = acc[:, :self.rank], acc[:, self.rank:]
        delta = delta[:, :self.rank], delta[:, self.rank:]
        tim1 = tim1[:, :self.rank], tim1[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        right = right[:, :, :self.rank], right[:, :, self.rank:]

        path1 = torch.cat([acc[0],delta[0],inv[0]],1),torch.cat([acc[1],delta[1],inv[1]],1)
        rel_path1 = self.out(F.relu(self.hidden1(path1[0]))),self.out(F.relu(self.hidden1(path1[1])))
        eng1 = rel_path1[0] - tran[0], rel_path1[1] - tran[1]

        user2 = self.embeddings[10](path2_user)
        path2_origin = self.proj_user(self.embeddings[0](path2_origin),user2)
        path2_mid = self.proj_user(self.embeddings[0](path2_mid),user2)
        path2_tran = self.proj_user(self.embeddings[1](x[:,8][0].repeat(len(path2_origin))),user2)
        path2_right = self.proj_users(self.embeddings[2].weight,user2)
        path2_origin = path2_origin[:, :self.rank], path2_origin[:, self.rank:]
        path2_mid = path2_mid[:, :self.rank], path2_mid[:, self.rank:]
        path2_tran = path2_tran[:, :self.rank], path2_tran[:, self.rank:]
        path2_right = path2_right[:, :,:self.rank], path2_right[:,:, self.rank:]

        path2 = torch.cat([path2_tran[0],path2_tran[0]],1),torch.cat([path2_tran[1],path2_tran[1]],1)
        rel_path2 = self.out(F.relu(self.hidden2(path2[0]))),self.out(F.relu(self.hidden2(path2[1])))
        eng2 = rel_path2[0] - path2_tran[0], rel_path2[1] - path2_tran[1]

        if not self.test_transition:
            score = (torch.squeeze(torch.bmm((tim[0] * inv[0] - tim[1] * inv[1]).view(-1,1,self.rank), torch.transpose(right[0],1,2))) +
             torch.squeeze(torch.bmm((tim[0] * inv[1] + tim[1] * inv[0]).view(-1,1,self.rank), torch.transpose(right[1],1,2))))
        else:
            score = (torch.squeeze(torch.bmm((origin[0] * tran[0] - origin[1] * tran[1]).view(-1,1,self.rank), torch.transpose(right[0],1,2))) +
             torch.squeeze(torch.bmm((origin[0] * tran[1] + origin[1] * tran[0]).view(-1,1,self.rank), torch.transpose(right[1],1,2))))

        user1 = self.embeddings[10](similar[:, 0])
        user1 = nn.functional.normalize(user1,dim=1)
        sim = self.embeddings[11](similar[:, 1])
        right_sim = self.embeddings[10].weight
        user1 = user1[:, :self.rank], user1[:, self.rank:]
        sim = sim[:, :self.rank], sim[:, self.rank:]
        right_sim = right_sim[:, :self.rank], right_sim[:, self.rank:]

        sim_score = (user1[0] * sim[0] - user1[1] * sim[1]) @ right_sim[0].t() + (user1[1] * sim[0] + user1[0] * sim[1]) @ right_sim[1].t()


        return score, sim_score,(
             torch.squeeze(torch.bmm((origin[0] * rel_path1[0] - origin[1] * rel_path1[1]).view(-1,1,self.rank), torch.transpose(right[0],1,2))) +
             torch.squeeze(torch.bmm((origin[0] * rel_path1[1] + origin[1] * rel_path1[0]).view(-1,1,self.rank), torch.transpose(right[1],1,2))) ),(
            torch.squeeze(torch.bmm((path2_origin[0] * rel_path2[0] - path2_origin[1] * rel_path2[1]).view(-1,1,self.rank), torch.transpose(path2_right[0],1,2))) +
             torch.squeeze(torch.bmm((path2_origin[0] * rel_path2[1] + path2_origin[1] * rel_path2[0]).view(-1,1,self.rank), torch.transpose(path2_right[1],1,2))) ),(
                     
                   torch.sqrt(origin[0] ** 2 + origin[1] ** 2),
                   torch.sqrt(tran[0] ** 2 + tran[1] ** 2),
                   torch.sqrt(acc[0] ** 2 + acc[1] ** 2),
                   torch.sqrt(inv[0] ** 2 + inv[1] ** 2),
                   torch.sqrt(loc[0] ** 2 + loc[1] ** 2),
                   torch.sqrt(delta[0] ** 2 + delta[1] ** 2),
                   torch.sqrt(tim1[0] ** 2 + tim1[1] ** 2),
                   torch.sqrt(tim[0] ** 2 + tim[1] ** 2),

               ), (torch.sum(torch.sqrt(eng1[0] ** 2 + eng1[1] ** 2)) / len(eng1) ), ((torch.sum(torch.sqrt(eng2[0] ** 2 + eng2[1] ** 2)) / len(eng2) ))


    def forward1(self, x):
        origin = self.embeddings[0](x[:, 6])
        loc = self.embeddings[2](x[:, 3])
        inv = self.embeddings[3](x[:, 5])
        acc = self.embeddings[4](x[:, 2])
        delta = self.diachronic(x[:, 7])
        tim1 = self.embeddings[8](x[:, 4])
        tim = self.embeddings[9](x[:, 1])
        user = self.embeddings[10](x[:, 0])
        user = nn.functional.normalize(user,dim=1)
        
        origin = self.proj_user(origin,user)
        loc = self.proj_user(loc,user)
        inv = self.proj_user(inv,user)
        acc = self.proj_user(acc,user)
        delta = self.proj_user(delta,user)
        tim1 = self.proj_user(tim1,user)
        tim = self.proj_user(tim,user)
        
        right_ots = self.proj_users(self.embeddings[8].weight,user)
        right_ts = self.proj_users(self.embeddings[9].weight,user)
        right_loc = self.proj_users(self.embeddings[2].weight,user)

        origin = origin[:, :self.rank], origin[:, self.rank:]
        loc = loc[:, :self.rank], loc[:, self.rank:]
        inv = inv[:, :self.rank], inv[:, self.rank:]
        acc = acc[:, :self.rank], acc[:, self.rank:]
        delta = delta[:, :self.rank], delta[:, self.rank:]
        tim1 = tim1[:, :self.rank], tim1[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        right_ots = right_ots[:, :, :self.rank], right_ots[:, :, self.rank:]
        right_ts = right_ts[:, :, :self.rank], right_ts[:, :, self.rank:]
        right_loc = right_loc[:, :, :self.rank], right_loc[:, :, self.rank:]

        return (torch.squeeze(torch.bmm((origin[0] * acc[0] - origin[1] * acc[1]).view(-1,1,self.rank), torch.transpose(right_ots[0],1,2))) +
             torch.squeeze(torch.bmm((origin[0] * acc[1] + origin[1] * acc[0]).view(-1,1,self.rank), torch.transpose(right_ots[1],1,2)))), (
             torch.squeeze(torch.bmm((tim1[0] * delta[0] - tim1[1] * delta[1]).view(-1,1,self.rank), torch.transpose(right_ts[0],1,2))) +
             torch.squeeze(torch.bmm((tim1[0] * delta[1] + tim1[1] * delta[0]).view(-1,1,self.rank), torch.transpose(right_ts[1],1,2))) ),(
             torch.squeeze(torch.bmm((tim[0] * inv[0] - tim[1] * inv[1]).view(-1,1,self.rank), torch.transpose(right_loc[0],1,2))) +
             torch.squeeze(torch.bmm((tim[0] * inv[1] + tim[1] * inv[0]).view(-1,1,self.rank), torch.transpose(right_loc[1],1,2)))
             )     
   

    def get_queries(self, queries: torch.Tensor):
        origin = self.embeddings[0](queries[:, 6])
        tran = self.embeddings[1](queries[:, 8])
        inv = self.embeddings[3](queries[:, 5])
        acc = self.embeddings[4](queries[:, 2])
        delta = self.diachronic(queries[:, 7])
        tim1 = self.embeddings[8](queries[:, 4])
        tim = self.embeddings[9](queries[:, 1])
        user = self.embeddings[10](queries[:, 0])

        origin = self.proj_user(origin,user)
        tran = self.proj_user(tran,user)
        inv = self.proj_user(inv,user)
        acc = self.proj_user(acc,user)
        delta = self.proj_user(delta,user)
        tim1 = self.proj_user(tim1,user)
        tim = self.proj_user(tim,user)
        locs = self.proj_users(self.embeddings[2].weight,user)

        origin = origin[:, :self.rank], origin[:, self.rank:]
        tran = tran[:, :self.rank], tran[:, self.rank:]

        inv = inv[:, :self.rank], inv[:, self.rank:]
        acc = acc[:, :self.rank], acc[:, self.rank:]
        delta = delta[:, :self.rank], delta[:, self.rank:]
        tim1 = tim1[:, :self.rank], tim1[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        locs = locs[:, :, :self.rank], locs[:, :, self.rank:]

        if not self.test_transition:
            score = (torch.squeeze(torch.bmm((tim[0] * inv[0] - tim[1] * inv[1]).view(-1,1,self.rank), torch.transpose(locs[0],1,2))) +
             torch.squeeze(torch.bmm((tim[0] * inv[1] + tim[1] * inv[0]).view(-1,1,self.rank), torch.transpose(locs[1],1,2))))
        else:
            score = (torch.squeeze(torch.bmm((origin[0] * tran[0] - origin[1] * tran[1]).view(-1,1,self.rank), torch.transpose(locs[0],1,2))) +
             torch.squeeze(torch.bmm((origin[0] * tran[1] + origin[1] * tran[0]).view(-1,1,self.rank), torch.transpose(locs[1],1,2))))

        return score

class MComplEx2(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int, int, int, int, int], rank: int,
            test_transition = True, init_size: float = 1e-3
    ):

        super(MComplEx2, self).__init__()
        self.sizes = sizes
        self.rank = rank
       
        self.time_nl = torch.sin
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2*rank, sparse=True)
            for s in [sizes[2], sizes[6], sizes[2], sizes[5], sizes[3], sizes[4], sizes[4], sizes[4], sizes[1], sizes[1],sizes[0],sizes[7]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size
        self.embeddings[4].weight.data *= init_size
        self.embeddings[5].weight.data *= init_size
        self.embeddings[6].weight.data *= init_size
        self.embeddings[7].weight.data *= init_size
        self.embeddings[8].weight.data *= init_size
        self.embeddings[9].weight.data *= init_size
        self.embeddings[10].weight.data *= init_size
        self.embeddings[11].weight.data *= init_size
        self.bn = nn.BatchNorm1d(sizes[2])
        self.bn = nn.BatchNorm1d(sizes[2])
        self.hidden1 = torch.nn.Linear(5*rank, 200)
        self.hidden2 = torch.nn.Linear(3*rank, 200)
        self.out = torch.nn.Linear(200, rank)
        self.test_transition = test_transition

       
    @staticmethod
    def has_time():
        return True

    def diachronic(self, x):
        delta_a = self.embeddings[5](x)
        delta_freq = self.embeddings[6](x)
        delta_phi = self.embeddings[7](x)
        delta = delta_a * self.time_nl(delta_freq * x.type(torch.FloatTensor).cuda().reshape((-1,1))  + delta_phi)
        return delta

    def proj_user(self, ent, user):
        ent = ent - user*torch.sum(user*ent,dim=1, keepdim=True)
        return ent

    def proj_users(self, pois, user):
        tmp = torch.mm(user,pois.transpose(0,1))
        pois = pois - torch.bmm(tmp[:,:,np.newaxis],user[:,np.newaxis,:])
        return pois

    
    def forward(self, x, path2_origin, path2_mid, path2_user, similar):
        origin = self.embeddings[0](x[:, 6])
        tran = self.embeddings[1](x[:, 8])
        loc = self.embeddings[2](x[:, 3])
        inv = self.embeddings[3](x[:, 5])
        acc = self.embeddings[4](x[:, 2])
        delta = self.diachronic(x[:, 7])
        tim1 = self.embeddings[8](x[:, 4])
        tim = self.embeddings[9](x[:, 1])
        user = self.embeddings[10](x[:, 0])
        
        origin = self.proj_user(origin,user)
        tran = self.proj_user(tran,user)
        loc = self.proj_user(loc,user)
        inv = self.proj_user(inv,user)
        acc = self.proj_user(acc,user)
        delta = self.proj_user(delta,user)
        tim1 = self.proj_user(tim1,user)
        tim = self.proj_user(tim,user)

        right = self.proj_users(self.embeddings[2].weight,user)


        origin = origin[:, :self.rank], origin[:, self.rank:]
        tran = tran[:, :self.rank], tran[:, self.rank:]
        loc = loc[:, :self.rank], loc[:, self.rank:]
        inv = inv[:, :self.rank], inv[:, self.rank:]
        acc = acc[:, :self.rank], acc[:, self.rank:]
        delta = delta[:, :self.rank], delta[:, self.rank:]
        tim1 = tim1[:, :self.rank], tim1[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        right = right[:, :, :self.rank], right[:, :, self.rank:]

        path1 = torch.cat([acc[0],tim1[0],delta[0],tim[0],inv[0]],1),torch.cat([acc[1],tim1[1],delta[1],tim[1],inv[1]],1)
        rel_path1 = self.out(F.relu(self.hidden1(path1[0]))),self.out(F.relu(self.hidden1(path1[1])))
        eng1 = rel_path1[0] - tran[0], rel_path1[1] - tran[1]

        user2 = self.embeddings[10](path2_user)
        path2_origin = self.proj_user(self.embeddings[0](path2_origin),user2)
        path2_mid = self.proj_user(self.embeddings[0](path2_mid),user2)
        path2_tran = self.proj_user(self.embeddings[1](x[:,8][0].repeat(len(path2_origin))),user2)
        path2_right = self.proj_users(self.embeddings[2].weight,user2)
        path2_origin = path2_origin[:, :self.rank], path2_origin[:, self.rank:]
        path2_mid = path2_mid[:, :self.rank], path2_mid[:, self.rank:]
        path2_tran = path2_tran[:, :self.rank], path2_tran[:, self.rank:]
        path2_right = path2_right[:, :,:self.rank], path2_right[:,:, self.rank:]

        path2 = torch.cat([path2_tran[0],path2_tran[0],path2_mid[0]],1),torch.cat([path2_tran[1],path2_tran[1],path2_mid[1]],1)
        rel_path2 = self.out(F.relu(self.hidden2(path2[0]))),self.out(F.relu(self.hidden2(path2[1])))
        eng2 = rel_path2[0] - path2_tran[0], rel_path2[1] - path2_tran[1]

        if not self.test_transition:
            score = (torch.squeeze(torch.bmm((tim[0] * inv[0] - tim[1] * inv[1]).view(-1,1,self.rank), torch.transpose(right[0],1,2))) +
             torch.squeeze(torch.bmm((tim[0] * inv[1] + tim[1] * inv[0]).view(-1,1,self.rank), torch.transpose(right[1],1,2))))
        else:
            score = (torch.squeeze(torch.bmm((origin[0] * tran[0] - origin[1] * tran[1]).view(-1,1,self.rank), torch.transpose(right[0],1,2))) +
             torch.squeeze(torch.bmm((origin[0] * tran[1] + origin[1] * tran[0]).view(-1,1,self.rank), torch.transpose(right[1],1,2))))

        user1 = self.embeddings[10](similar[:, 0])
        user1 = nn.functional.normalize(user1,dim=1)
        sim = self.embeddings[11](similar[:, 1])
        right_sim = self.embeddings[10].weight
        user1 = user1[:, :self.rank], user1[:, self.rank:]
        sim = sim[:, :self.rank], sim[:, self.rank:]
        right_sim = right_sim[:, :self.rank], right_sim[:, self.rank:]

        sim_score = (user1[0] * sim[0] - user1[1] * sim[1]) @ right_sim[0].t() + (user1[1] * sim[0] + user1[0] * sim[1]) @ right_sim[1].t()


        return score,sim_score, (
             torch.squeeze(torch.bmm((origin[0] * rel_path1[0] - origin[1] * rel_path1[1]).view(-1,1,self.rank), torch.transpose(right[0],1,2))) +
             torch.squeeze(torch.bmm((origin[0] * rel_path1[1] + origin[1] * rel_path1[0]).view(-1,1,self.rank), torch.transpose(right[1],1,2))) ),(
            torch.squeeze(torch.bmm((path2_origin[0] * rel_path2[0] - path2_origin[1] * rel_path2[1]).view(-1,1,self.rank), torch.transpose(path2_right[0],1,2))) +
             torch.squeeze(torch.bmm((path2_origin[0] * rel_path2[1] + path2_origin[1] * rel_path2[0]).view(-1,1,self.rank), torch.transpose(path2_right[1],1,2))) ),(
                     
                   torch.sqrt(origin[0] ** 2 + origin[1] ** 2),
                   torch.sqrt(tran[0] ** 2 + tran[1] ** 2),
                   torch.sqrt(acc[0] ** 2 + acc[1] ** 2),
                   torch.sqrt(inv[0] ** 2 + inv[1] ** 2),
                   torch.sqrt(loc[0] ** 2 + loc[1] ** 2),
                   torch.sqrt(delta[0] ** 2 + delta[1] ** 2),
                   torch.sqrt(tim1[0] ** 2 + tim1[1] ** 2),
                   torch.sqrt(tim[0] ** 2 + tim[1] ** 2),

               ), (torch.sum(torch.sqrt(eng1[0] ** 2 + eng1[1] ** 2)) / len(eng1) ), ((torch.sum(torch.sqrt(eng2[0] ** 2 + eng2[1] ** 2)) / len(eng2) ))


    def forward1(self, x):
        origin = self.embeddings[0](x[:, 6])
        loc = self.embeddings[2](x[:, 3])
        inv = self.embeddings[3](x[:, 5])
        acc = self.embeddings[4](x[:, 2])
        delta = self.diachronic(x[:, 7])
        tim1 = self.embeddings[8](x[:, 4])
        tim = self.embeddings[9](x[:, 1])
        user = self.embeddings[10](x[:, 0])
        user = nn.functional.normalize(user,dim=1)
        
        origin = self.proj_user(origin,user)
        loc = self.proj_user(loc,user)
        inv = self.proj_user(inv,user)
        acc = self.proj_user(acc,user)
        delta = self.proj_user(delta,user)
        tim1 = self.proj_user(tim1,user)
        tim = self.proj_user(tim,user)
        
        right_ots = self.proj_users(self.embeddings[8].weight,user)
        right_ts = self.proj_users(self.embeddings[9].weight,user)
        right_loc = self.proj_users(self.embeddings[2].weight,user)

        origin = origin[:, :self.rank], origin[:, self.rank:]
        loc = loc[:, :self.rank], loc[:, self.rank:]
        inv = inv[:, :self.rank], inv[:, self.rank:]
        acc = acc[:, :self.rank], acc[:, self.rank:]
        delta = delta[:, :self.rank], delta[:, self.rank:]
        tim1 = tim1[:, :self.rank], tim1[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        right_ots = right_ots[:, :, :self.rank], right_ots[:, :, self.rank:]
        right_ts = right_ts[:, :, :self.rank], right_ts[:, :, self.rank:]
        right_loc = right_loc[:, :, :self.rank], right_loc[:, :, self.rank:]

        return (torch.squeeze(torch.bmm((origin[0] * acc[0] - origin[1] * acc[1]).view(-1,1,self.rank), torch.transpose(right_ots[0],1,2))) +
             torch.squeeze(torch.bmm((origin[0] * acc[1] + origin[1] * acc[0]).view(-1,1,self.rank), torch.transpose(right_ots[1],1,2)))), (
             torch.squeeze(torch.bmm((tim1[0] * delta[0] - tim1[1] * delta[1]).view(-1,1,self.rank), torch.transpose(right_ts[0],1,2))) +
             torch.squeeze(torch.bmm((tim1[0] * delta[1] + tim1[1] * delta[0]).view(-1,1,self.rank), torch.transpose(right_ts[1],1,2))) ),(
             torch.squeeze(torch.bmm((tim[0] * inv[0] - tim[1] * inv[1]).view(-1,1,self.rank), torch.transpose(right_loc[0],1,2))) +
             torch.squeeze(torch.bmm((tim[0] * inv[1] + tim[1] * inv[0]).view(-1,1,self.rank), torch.transpose(right_loc[1],1,2)))
             )     
   

    def get_queries(self, queries: torch.Tensor):
        origin = self.embeddings[0](queries[:, 6])
        tran = self.embeddings[1](queries[:, 8])
        inv = self.embeddings[3](queries[:, 5])
        acc = self.embeddings[4](queries[:, 2])
        delta = self.diachronic(queries[:, 7])
        tim1 = self.embeddings[8](queries[:, 4])
        tim = self.embeddings[9](queries[:, 1])
        user = self.embeddings[10](queries[:, 0])

        origin = self.proj_user(origin,user)
        tran = self.proj_user(tran,user)
        inv = self.proj_user(inv,user)
        acc = self.proj_user(acc,user)
        delta = self.proj_user(delta,user)
        tim1 = self.proj_user(tim1,user)
        tim = self.proj_user(tim,user)
        locs = self.proj_users(self.embeddings[2].weight,user)

        origin = origin[:, :self.rank], origin[:, self.rank:]
        tran = tran[:, :self.rank], tran[:, self.rank:]

        inv = inv[:, :self.rank], inv[:, self.rank:]
        acc = acc[:, :self.rank], acc[:, self.rank:]
        delta = delta[:, :self.rank], delta[:, self.rank:]
        tim1 = tim1[:, :self.rank], tim1[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        locs = locs[:, :, :self.rank], locs[:, :, self.rank:]

        if not self.test_transition:
            score = (torch.squeeze(torch.bmm((tim[0] * inv[0] - tim[1] * inv[1]).view(-1,1,self.rank), torch.transpose(locs[0],1,2))) +
             torch.squeeze(torch.bmm((tim[0] * inv[1] + tim[1] * inv[0]).view(-1,1,self.rank), torch.transpose(locs[1],1,2))))
        else:
            score = (torch.squeeze(torch.bmm((origin[0] * tran[0] - origin[1] * tran[1]).view(-1,1,self.rank), torch.transpose(locs[0],1,2))) +
             torch.squeeze(torch.bmm((origin[0] * tran[1] + origin[1] * tran[0]).view(-1,1,self.rank), torch.transpose(locs[1],1,2))))

        return score

class MComplEx22(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int, int, int, int, int], rank: int,
            test_transition = True, init_size: float = 1e-3
    ):

        super(MComplEx22, self).__init__()
        self.sizes = sizes
        self.rank = rank
       
        self.time_nl = torch.sin
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2*rank, sparse=True)
            for s in [sizes[2], sizes[6], sizes[2], sizes[5], sizes[3], sizes[4], sizes[4], sizes[4], sizes[1], sizes[1],sizes[0],sizes[7]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size
        self.embeddings[4].weight.data *= init_size
        self.embeddings[5].weight.data *= init_size
        self.embeddings[6].weight.data *= init_size
        self.embeddings[7].weight.data *= init_size
        self.embeddings[8].weight.data *= init_size
        self.embeddings[9].weight.data *= init_size
        self.embeddings[10].weight.data *= init_size
        self.embeddings[11].weight.data *= init_size
        self.bn = nn.BatchNorm1d(sizes[2])
        self.bn = nn.BatchNorm1d(sizes[2])
        self.hidden1 = torch.nn.Linear(5*rank, 200)
        self.hidden2 = torch.nn.Linear(3*rank, 200)
        self.out = torch.nn.Linear(200, rank)
        self.test_transition = test_transition

       
    @staticmethod
    def has_time():
        return True

    def diachronic(self, x):
        delta_a = self.embeddings[5](x)
        delta_freq = self.embeddings[6](x)
        delta_phi = self.embeddings[7](x)
        delta = delta_a * self.time_nl(delta_freq * x.type(torch.FloatTensor).cuda().reshape((-1,1))  + delta_phi)
        return delta

    def proj_user(self, ent, user):
        ent = ent - user*torch.sum(user*ent,dim=1, keepdim=True)
        return ent

    def proj_users(self, pois, user):
        tmp = torch.mm(user,pois.transpose(0,1))
        pois = pois - torch.bmm(tmp[:,:,np.newaxis],user[:,np.newaxis,:])
        return pois

    
    def forward(self, x, path2_origin, path2_mid, path2_user, similar):
        origin = self.embeddings[0](x[:, 6])
        tran = self.embeddings[1](x[:, 8])
        loc = self.embeddings[2](x[:, 3])
        inv = self.embeddings[3](x[:, 5])
        acc = self.embeddings[4](x[:, 2])
        delta = self.diachronic(x[:, 7])
        tim1 = self.embeddings[8](x[:, 4])
        tim = self.embeddings[9](x[:, 1])
        # user = self.embeddings[10](x[:, 0])
        
        # origin = self.proj_user(origin,user)
        # tran = self.proj_user(tran,user)
        # loc = self.proj_user(loc,user)
        # inv = self.proj_user(inv,user)
        # acc = self.proj_user(acc,user)
        # delta = self.proj_user(delta,user)
        # tim1 = self.proj_user(tim1,user)
        # tim = self.proj_user(tim,user)

        # right = self.proj_users(self.embeddings[2].weight,user)
        right = self.embeddings[2].weight


        origin = origin[:, :self.rank], origin[:, self.rank:]
        tran = tran[:, :self.rank], tran[:, self.rank:]
        loc = loc[:, :self.rank], loc[:, self.rank:]
        inv = inv[:, :self.rank], inv[:, self.rank:]
        acc = acc[:, :self.rank], acc[:, self.rank:]
        delta = delta[:, :self.rank], delta[:, self.rank:]
        tim1 = tim1[:, :self.rank], tim1[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        right = right[:, :self.rank], right[:, self.rank:]

        path1 = torch.cat([acc[0],tim1[0],delta[0],tim[0],inv[0]],1),torch.cat([acc[1],tim1[1],delta[1],tim[1],inv[1]],1)
        rel_path1 = self.out(F.relu(self.hidden1(path1[0]))),self.out(F.relu(self.hidden1(path1[1])))
        eng1 = rel_path1[0] - tran[0], rel_path1[1] - tran[1]

        # user2 = self.embeddings[10](path2_user)
        # path2_origin = self.proj_user(self.embeddings[0](path2_origin),user2)
        # path2_mid = self.proj_user(self.embeddings[0](path2_mid),user2)
        # path2_tran = self.proj_user(self.embeddings[1](x[:,8][0].repeat(len(path2_origin))),user2)
        # path2_right = self.proj_users(self.embeddings[2].weight,user2)
        path2_origin = self.embeddings[0](path2_origin)
        path2_mid = self.embeddings[0](path2_mid)
        path2_tran  = self.embeddings[1](x[:,8][0].repeat(len(path2_origin)))
        # path2_right  = self.embeddings[2].weight

        path2_origin = path2_origin[:, :self.rank], path2_origin[:, self.rank:]
        path2_mid = path2_mid[:, :self.rank], path2_mid[:, self.rank:]
        path2_tran = path2_tran[:, :self.rank], path2_tran[:, self.rank:]
        # path2_right = path2_right[:,:self.rank], path2_right[:, self.rank:]

        path2 = torch.cat([path2_tran[0],path2_tran[0],path2_mid[0]],1),torch.cat([path2_tran[1],path2_tran[1],path2_mid[1]],1)
        rel_path2 = self.out(F.relu(self.hidden2(path2[0]))),self.out(F.relu(self.hidden2(path2[1])))
        eng2 = rel_path2[0] - path2_tran[0], rel_path2[1] - path2_tran[1]

        if not self.test_transition:
            score = (tim[0] * inv[0] - tim[1] * inv[1]) @ right[0].t() + (tim[1] * inv[0] + tim[0] * inv[1]) @ right[1].t()
        else:
            score = (origin[0] * tran[0] - origin[1] * tran[1]) @ right[0].t() + (origin[1] * tran[0] + origin[0] * tran[1]) @ right[1].t()
        
        user1 = self.embeddings[10](similar[:, 0])
        user1 = nn.functional.normalize(user1,dim=1)
        sim = self.embeddings[11](similar[:, 1])
        right_sim = self.embeddings[10].weight
        user1 = user1[:, :self.rank], user1[:, self.rank:]
        sim = sim[:, :self.rank], sim[:, self.rank:]
        right_sim = right_sim[:, :self.rank], right_sim[:, self.rank:]

        sim_score = (user1[0] * sim[0] - user1[1] * sim[1]) @ right_sim[0].t() + (user1[1] * sim[0] + user1[0] * sim[1]) @ right_sim[1].t()


        return score, sim_score,(
             (origin[0] * rel_path1[0] - origin[1] * rel_path1[1]) @ right[0].t() + (origin[1] * rel_path1[0] + origin[0] * rel_path1[1]) @ right[1].t()),(
                 (path2_origin[0] * rel_path2[0] - path2_origin[1] * rel_path2[1]) @ right[0].t() + (path2_origin[1] * rel_path2[0] + path2_origin[0] * rel_path2[1]) @ right[1].t()),(
                     
                   torch.sqrt(origin[0] ** 2 + origin[1] ** 2),
                   torch.sqrt(tran[0] ** 2 + tran[1] ** 2),
                   torch.sqrt(acc[0] ** 2 + acc[1] ** 2),
                   torch.sqrt(inv[0] ** 2 + inv[1] ** 2),
                   torch.sqrt(loc[0] ** 2 + loc[1] ** 2),
                   torch.sqrt(delta[0] ** 2 + delta[1] ** 2),
                   torch.sqrt(tim1[0] ** 2 + tim1[1] ** 2),
                   torch.sqrt(tim[0] ** 2 + tim[1] ** 2),
                   ), (torch.sum(torch.sqrt(eng1[0] ** 2 + eng1[1] ** 2)) / len(eng1) ), (
                       torch.sum(torch.sqrt(eng2[0] ** 2 + eng2[1] ** 2)) / len(eng2) )

    def forward1(self, x):
        origin = self.embeddings[0](x[:, 6])
        loc = self.embeddings[2](x[:, 3])
        inv = self.embeddings[3](x[:, 5])
        acc = self.embeddings[4](x[:, 2])
        delta = self.diachronic(x[:, 7])
        tim1 = self.embeddings[8](x[:, 4])
        tim = self.embeddings[9](x[:, 1])
        # user = self.embeddings[10](x[:, 0])
        # user = nn.functional.normalize(user,dim=1)
        
        # origin = self.proj_user(origin,user)
        # loc = self.proj_user(loc,user)
        # inv = self.proj_user(inv,user)
        # acc = self.proj_user(acc,user)
        # delta = self.proj_user(delta,user)
        # tim1 = self.proj_user(tim1,user)
        # tim = self.proj_user(tim,user)
        
        # right_ots = self.proj_users(self.embeddings[8].weight,user)
        # right_ts = self.proj_users(self.embeddings[9].weight,user)
        # right_loc = self.proj_users(self.embeddings[2].weight,user)
        right_ots = self.embeddings[8].weight
        right_ts = self.embeddings[9].weight
        right_loc = self.embeddings[2].weight

        origin = origin[:, :self.rank], origin[:, self.rank:]
        loc = loc[:, :self.rank], loc[:, self.rank:]
        inv = inv[:, :self.rank], inv[:, self.rank:]
        acc = acc[:, :self.rank], acc[:, self.rank:]
        delta = delta[:, :self.rank], delta[:, self.rank:]
        tim1 = tim1[:, :self.rank], tim1[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        right_ots = right_ots[:,  :self.rank], right_ots[:, self.rank:]
        right_ts = right_ts[:, :self.rank], right_ts[:, self.rank:]
        right_loc = right_loc[:,:self.rank], right_loc[:, self.rank:]

        return ((origin[0] * acc[0] - origin[1] * acc[1]) @ right_ots[0].t() + (origin[1] * acc[0] + origin[0] * acc[1]) @ right_ots[1].t()), (
             (tim1[0] * delta[0] - tim1[1] * delta[1]) @ right_ts[0].t() + (tim1[1] * delta[0] + tim1[0] * delta[1]) @ right_ts[1].t() ),(
            (tim[0] * inv[0] - tim[1] * inv[1]) @ right_loc[0].t() + (tim[1] * inv[0] + tim[0] * inv[1]) @ right_loc[1].t())     

  


    def get_queries(self, queries: torch.Tensor):
        origin = self.embeddings[0](queries[:, 6])
        tran = self.embeddings[1](queries[:, 8])
        inv = self.embeddings[3](queries[:, 5])
        acc = self.embeddings[4](queries[:, 2])
        delta = self.diachronic(queries[:, 7])
        tim1 = self.embeddings[8](queries[:, 4])
        tim = self.embeddings[9](queries[:, 1])
        # user = self.embeddings[10](queries[:, 0])

        # origin = self.proj_user(origin,user)
        # tran = self.proj_user(tran,user)
        # inv = self.proj_user(inv,user)
        # acc = self.proj_user(acc,user)
        # delta = self.proj_user(delta,user)
        # tim1 = self.proj_user(tim1,user)
        # tim = self.proj_user(tim,user)
        # locs = self.proj_users(self.embeddings[2].weight,user)
        locs = self.embeddings[2].weight

        origin = origin[:, :self.rank], origin[:, self.rank:]
        tran = tran[:, :self.rank], tran[:, self.rank:]

        inv = inv[:, :self.rank], inv[:, self.rank:]
        acc = acc[:, :self.rank], acc[:, self.rank:]
        delta = delta[:, :self.rank], delta[:, self.rank:]
        tim1 = tim1[:, :self.rank], tim1[:, self.rank:]
        tim = tim[:, :self.rank], tim[:, self.rank:]
        locs = locs[:, :self.rank], locs[:, self.rank:]

        if not self.test_transition:
            score = (tim[0] * inv[0] - tim[1] * inv[1]) @ locs[0].t() + (tim[1] * inv[0] + tim[0] * inv[1]) @ locs[1].t()
        else:
            score = (origin[0] * tran[0] - origin[1] * tran[1]) @ locs[0].t() + (origin[1] * tran[0] + origin[0] * tran[1]) @ locs[1].t()
   


        return score

