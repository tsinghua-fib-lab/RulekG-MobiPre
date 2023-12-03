# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
from typing import Dict
import logging
import torch
from torch import optim
import numpy as np
from collections import defaultdict
from datasets import TemporalDataset
from optimizers import TKBCOptimizer
from models import MComplEx,MComplEx2,RComplEx,MComplEx11,MComplEx12,ComplEx,MComplEx21,MComplEx22
from regularizers import N3, Lambda3
import setproctitle
import os
import time
from torch.optim.lr_scheduler import ExponentialLR
setproctitle.setproctitle("rule@yuqh")
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

parser = argparse.ArgumentParser(
    description="Temporal ComplEx"
)
parser.add_argument(
    '--dataset', type=str,
    help="Dataset name"
)
models = [
      'MComplEx','MComplEx2','RComplEx','MComplEx11','MComplEx12','ComplEx','MComplEx21','MComplEx22'
]
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)
parser.add_argument(
    '--max_epochs', default=400, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid_freq', default=5, type=int,
    help="Number of epochs between each valid."
)
parser.add_argument(
    '--rank', default=40, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=64, type=int,
    help="Batch size."
)
parser.add_argument(
    '--learning_rate', default=0.05, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--emb_reg', default=0.01, type=float,
    help="Embedding regularizer strength"
)
parser.add_argument(
    '--triple_reg', default=0.01, type=float,
    help="Timestamp regularizer strength"
)
parser.add_argument(
    '--user_ratio', default=0.5, type=float,
    help="dymatic vector length ratio"
)
parser.add_argument(
    '--test_transition', default=True,action="store_true",
    help="Use a specific embedding for category relations"
)
args = parser.parse_args()

dataset = TemporalDataset(args.dataset)
sizes = dataset.get_shape()

print(sizes)
model = {
    'ComplEx': ComplEx(sizes,  args.rank,test_transition = args.test_transition),
     'RComplEx': RComplEx(sizes, args.rank,args.user_ratio, test_transition = args.test_transition),
    'MComplEx': MComplEx(sizes, args.rank, test_transition = args.test_transition),
    'MComplEx11': MComplEx11(sizes, args.rank, test_transition = args.test_transition),
    'MComplEx12': MComplEx12(sizes, args.rank,  test_transition = args.test_transition),
    'MComplEx2': MComplEx2(sizes, args.rank, test_transition = args.test_transition),
    'MComplEx21': MComplEx21(sizes, args.rank,  test_transition = args.test_transition),
    'MComplEx22': MComplEx22(sizes, args.rank,  test_transition = args.test_transition),
    
}[args.model]
model = model.cuda()

torch.backends.cudnn.deterministic = True
seed = 30
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

opt = optim.Adagrad(model.parameters(), lr=args.learning_rate)

# opt = optim.Adam(model.parameters(), lr = args.learning_rate)
# scheduler = ExponentialLR(opt, args.decay_rate)

emb_reg = N3(args.emb_reg)
triple_reg = Lambda3(args.triple_reg)

test_result = []
MRR_all=[]
for epoch in range(args.max_epochs):
    examples = torch.from_numpy(
        dataset.get_train1().astype('int64')
    )
    confidence1 = torch.from_numpy(
        dataset.get_confidence1().astype('int64')
    )
    similar = torch.from_numpy(
        dataset.get_similar().astype('int64')
    )
    examples2 = [i for i in dataset.get_train2()]

    model.train()
    optimizer = TKBCOptimizer(
        model, emb_reg, triple_reg, opt,model_name = args.model,
        batch_size=args.batch_size, n_predictions = sizes[2]
    )
    optimizer.epoch(examples,examples2,similar)
   
    def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
        """
        aggregate metrics for missing lhs and rhs
        :param mrrs: d
        :param hits:
        :return:
        """
        m = mrrs['loc']
        h = hits['loc']
        return {'MRR': m, 'hits@[1,5,10]': h}

    if epoch < 0 or (epoch + 1) % args.valid_freq == 0:
        valid, test, train = [
            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 500000000000000))
            for split in ['valid', 'test', 'train']
        ]
        print("valid(MRR): ", valid['MRR'])
        print("test: ", test['MRR'])
        print("train: ", train['MRR'])
        print("valid(hits): ", valid['hits@[1,5,10]'])
        print("test: ", test['hits@[1,5,10]'])
        print("train: ", train['hits@[1,5,10]'])
        test_result.append([test['MRR'],test['hits@[1,5,10]']])
        MRR_all.append(float(valid['MRR']))
    if len(MRR_all)>3:
        if (MRR_all[-1]-MRR_all[-2])<0 and (MRR_all[-2]-MRR_all[-3])<0 and (MRR_all[-3]-MRR_all[-4])<0 or epoch == args.max_epochs -1:
            index = MRR_all.index(max(MRR_all))
            print(test_result[index])
            break
print(test_result)
   
