
from numpy.core.fromnumeric import repeat
from torch import tensor
import tqdm
import torch
from torch import nn
from torch import optim
import numpy as np
from models import TKBCModel
from regularizers import Regularizer
from datasets import TemporalDataset
import random
import torch.nn.functional as F
from collections import defaultdict

class TKBCOptimizer(object):
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, triple_regularizer: Regularizer,
            optimizer: optim.Optimizer, model_name : str, batch_size: int = 64, n_predictions: int =32,
            
            verbose: bool = True
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.triple_regularizer = triple_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        # self.scheduler = scheduler
        self.n_predictions = n_predictions
        self.model_name = model_name

    def epoch(self, examples: torch.LongTensor, examples2 , similar: torch.LongTensor):
        ranks = torch.randperm(examples.shape[0])
        actual_examples = examples[ranks, :]
        actual_examples2 = [examples2[i] for i in ranks.tolist()]
        loss = nn.CrossEntropyLoss(reduction='mean')
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
           
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[b_begin:b_begin + self.batch_size].cuda()
                input_batch2 = actual_examples2[b_begin:b_begin + self.batch_size]
                truth = input_batch[:, 3] 
                similar_truth = similar.cuda()[:,2]

                path2_user = []
                origin_loc2 = []
                for line in input_batch2:
                    if len(line)>3:
                        path2_user += [line[0]]*(int(((len(line)-3) * (len(line)-2))/2))
                        for i in range(1,len(line)-2):
                            for j in range(i+1,len(line)-1):
                                origin_loc2.append([line[i],line[j],line[-1]])

                
                path2_user = torch.tensor(path2_user).cuda()
                path2_origin = torch.tensor(np.array(origin_loc2)[:,0]).cuda()
                path2_mid = torch.tensor(np.array(origin_loc2)[:,1]).cuda()
                path2_truth = torch.tensor(np.array(origin_loc2)[:,2]).cuda()

                if self.model_name in ['MComplEx','RComplEx']:
                    predictions, similar_predictions, factors = self.model.forward(input_batch, similar.cuda())

                elif self.model_name in ['MComplEx11']:
                    predictions, similar_predictions, path_predictions, factors, l_energy = self.model.forward(input_batch,similar.cuda())
                    l_path = loss(path_predictions,truth)
                elif self.model_name in ['MComplEx12']:
                    predictions, similar_predictions, path_predictions, factors, l_energy = self.model.forward(input_batch,path2_origin,path2_mid,path2_user,similar.cuda())
                    l_path = loss(path_predictions,path2_truth)
                else:
                    predictions, similar_predictions, path1_predictions,path2_predictions, factors, l_energy1, l_energy2 = self.model.forward(input_batch,path2_origin,path2_mid,path2_user,similar.cuda())
                    l_energy = l_energy1 + l_energy2
                    l_path1 = loss(path1_predictions,truth)
                    # l_path2 = loss(path2_predictions,path2_truth)
                l_sim = loss(similar_predictions,similar_truth)

                a1 = b1 =1
                a2 = b2 =1
                c=0.1

                l_fit = loss(predictions, truth) 
                l_reg = self.emb_regularizer.forward(factors)
                p_acc, p_delta, p_inv = self.model.forward1(input_batch)
                l_tri = loss(p_acc,input_batch[:, 4]) + loss(p_delta,input_batch[:, 1]) + loss(p_inv,input_batch[:, 3])
                # l_tri = torch.tensor(0.0)

                if self.model_name in ['MComplEx','RComplEx']:
                    l = (l_fit  + l_reg +  c *l_tri)
                elif self.model_name in ['MComplEx11','MComplEx12']:
                    l = (l_fit + a1 * l_path + b1 * l_energy + l_reg +  c *l_tri )
                else:
                    l = (l_fit + a1 * l_path1 + l_reg + c * l_tri )
                    
                l = l
                self.optimizer.zero_grad()
                l.backward()
                
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(
                    loss=f'{l_fit.item():.0f}',
                    # l_energy1 =f'{l_energy1.item():.0f}',
                    reg=f'{l_reg.item():.0f}',
                    tri=f'{l_tri.item()*c:.0f}',
                    sim=f'{l_sim.item():.0f}',
                )
            # self.scheduler.step()


