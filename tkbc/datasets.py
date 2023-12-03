# Copyright (c) Facebook, Inc. and its affiliates.
#

from pathlib import Path
import pkg_resources
import pickle
from collections import defaultdict
from typing import Dict, Tuple, List

from sklearn.metrics import average_precision_score
from datetime import datetime 

import numpy as np
import torch
from models import TKBCModel

DATA_PATH = './data1'
#DATA_PATH = pkg_resources.resource_filename('tkbc', 'data/')

class TemporalDataset(object):
    def __init__(self, name: str):
        self.root = Path(DATA_PATH) / name

        self.data = {}
        for f in ['train', 'test', 'valid', 'numbers','confidence1','train2','test2','valid2','similar']:
            in_file = open(str(self.root / (f + '.pickle')), 'rb')
            self.data[f] = pickle.load(in_file)
        numbers = self.data['numbers']

        # len(users), len(pois),len(timestamps),len(access), len(inverse),max(deltats), len(transfer)
        self.n_ids = int(numbers[0])
        self.n_pois = int(numbers[1])
        self.n_timestamps = int(numbers[2])
        self.n_access = int(numbers[3])
        self.n_inverse = int(numbers[4])
        self.n_deltat = int(numbers[5] + 1)
        self.n_transfer = int(numbers[6])
        self.n_sim = int(numbers[3])
        
        
        print("Assume all timestamps are regularly spaced")
        print("Not using time intervals and events eval")
        self.events = None

    def has_intervals(self):
        return self.events is not None

    def get_examples(self, split):
        return self.data[split]

    def get_train1(self):
        copy = np.copy(self.data['train'])
        return copy

    def get_train2(self):
        copy = np.copy(self.data['train2'])
        return copy

    def get_confidence1(self):
        copy = np.copy(self.data['confidence1'])
        return copy
    
    def get_similar(self):
        return self.data['similar']
 
    def eval(
            self, model: TKBCModel, split: str, n_queries: int = -1, missing_eval: str = 'loc',
            at: Tuple[int] = (1, 5, 10)
    ):  
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64')).cuda()
        mean_reciprocal_rank = {}
        hits_at = {}

        q = examples.clone()
        permutation = torch.randperm(len(examples))[:n_queries]
        q = examples[permutation]
        m = missing_eval
        ranks, user_rank = model.get_ranking(q, split, batch_size=300)
        user_ranks = {}
        for i in range(len(user_rank)):
            if user_rank[i] not in user_ranks:
                user_ranks[user_rank[i]] = [ranks[i].item()]  
            else:
                user_ranks[user_rank[i]] += [ranks[i].item()]
        
        reciprocal_rank_users = {}
        hits_users = {}
        for i in user_ranks.keys():
            user_num = len(user_ranks[i])
            reciprocal_rank_users[i] = np.sum(1. / np.array(user_ranks[i]))/user_num
            hits_users[i]=[]
            for j in at:
                hits1 = [rank for rank in user_ranks[i] if rank <=j]
                hits_users[i] +=[len(hits1)/user_num]
        
        mean_reciprocal_rank[m] = format(np.mean(list(reciprocal_rank_users.values())),'.4f')
        #torch.mean(1. / ranks).item()
        hits_at[m] = [float(format(i,'.4f')) for i in np.mean(np.array(list(hits_users.values())),axis = 0)]
    
        return mean_reciprocal_rank, hits_at
          

    def get_shape(self):
        return self.n_ids, self.n_timestamps, self.n_pois, self.n_access,  self.n_deltat, self.n_inverse, self.n_transfer,self.n_sim


        