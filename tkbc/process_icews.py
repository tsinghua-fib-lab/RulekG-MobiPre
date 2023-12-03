# Copyright (c) Facebook, Inc. and its affiliates.

import pkg_resources
import os
import errno
from pathlib import Path
import pickle
import random
import numpy as np
import time
import shutil
from collections import defaultdict,Counter
import setproctitle
setproctitle.setproctitle("location-yuqh")
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#DATA_PATH = pkg_resources.resource_filename('tkbc', 'data/')
DATA_PATH='./data1/'


cat1=open('./data/Relation_beijing_poi_1_CatOf.txt',encoding='utf-8')
cs1={}
for line in cat1:
    tmp=line.strip().split('\t')
    cs1[tmp[0]]=tmp[1]

cat2=open('./data/Relation_beijing_poi_2_CatOf.txt',encoding='utf-8')
cs2={}
for line in cat2:
    tmp=line.strip().split('\t')
    cs2[tmp[0]]=tmp[1] 

cat3=open('./data/Relation_beijing_poi_3_CatOf.txt',encoding='utf-8')
cs3={}
for line in cat3:
    tmp=line.strip().split('\t')
    cs3[tmp[0]]=tmp[1] 
    

def tid_96(ts):
    tm = time.strptime(ts, "%Y%m%d%H%M")
    if tm.tm_wday in [0, 1, 2, 3, 4]:
        t_hour = tm.tm_hour * 2 
    else:
        t_hour = (tm.tm_hour + 24) * 2
    if tm.tm_min < 30:
        tid = t_hour
    else:
        tid = t_hour + 1
    return str(tid)

                
def prepare_dataset(path, name):

    source_file = open('./data/pois.txt', 'r').read().strip().split('\n')
    print('records:',len(source_file))
    sort_users = defaultdict(list)
    for line in source_file:
        if len(line.split(',')) ==3:
            user,poi,ts = eval(line)
            sort_users[user].append([user,poi,ts])
 
    train2 = []
    test2 = []
    valid2 = []
    for user in list(sort_users.keys())[:500]:
        traces = sort_users[user]
        for i in range(1,int((len(sort_users[user])-1)*0.7)+1):
            times = set()
            for j in range(i):
                if time.mktime(time.strptime(traces[j][2], "%Y%m%d%H%M"))+ 25*60*60 > time.mktime(time.strptime(traces[i][2], "%Y%m%d%H%M")):
                   times.add(traces[j][1])
                times.add(traces[i-1][1])
            train2.append([user] + list(times)+[traces[i][1]])
        for i in range(int((len(sort_users[user])-1)*0.7)+1,int((len(sort_users[user])-1)*0.8)+1):
            times = set()
            for j in range(i):
                if time.mktime(time.strptime(traces[j][2], "%Y%m%d%H%M"))+ 25*60*60 > time.mktime(time.strptime(traces[i][2], "%Y%m%d%H%M")):
                   times.add(traces[j][1]) 
                times.add(traces[i-1][1])
            valid2.append([user] + list(times)+[traces[i][1]])
        for i in range(int((len(sort_users[user])-1)*0.8)+1,len(sort_users[user])):
            times = set()
            for j in range(i):
                if time.mktime(time.strptime(traces[j][2], "%Y%m%d%H%M"))+ 25*60*60 > time.mktime(time.strptime(traces[i][2], "%Y%m%d%H%M")):
                   times.add(traces[j][1]) 
                times.add(traces[i-1][1])
            test2.append([user] + list(times)+[traces[i][1]])

    sort_users1 = defaultdict(list)
    for user in list(sort_users.keys())[:500]:
        for i in range(len(sort_users[user])-1):
            new_line = sort_users[user][i] + sort_users[user][i+1]
            sort_users1[user].append(new_line)

    sort_users = sort_users1.copy()
    train1 = []
    valid1 = []
    test1 = []
    for k in sort_users.keys():
        train1 += sort_users[k][0:int(0.7*len(sort_users[k]))]
        valid1 += sort_users[k][int(0.7*len(sort_users[k])):int(0.8*len(sort_users[k]))]
        test1 += sort_users[k][int(0.8*len(sort_users[k])):]

    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    # write ent to id / rel to id
    for (dic, f) in zip([train1,valid1,test1, train2, valid2,test2], ['train1','valid1','test1','train2','valid2','test2']):
        ff = open(os.path.join(path, f), 'w+')
        for line in dic:
            ff.write(str(line)+"\n")
        ff.close()
    
    files = ['train1', 'valid1', 'test1']
    users, access, transfer,inverse, pois,timestamps, deltats, sim = set(), set(),set(), set(), set(), set(),set(), set()
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            _, origin1, timestamp1, user, loc, timestamp = eval(line)
            ts = tid_96(timestamp)
            ts1 = time.mktime(time.strptime(timestamp1, "%Y%m%d%H%M"))
            ts2 = time.mktime(time.strptime(timestamp, "%Y%m%d%H%M"))
            deltat =  int((ts2-ts1)//3600)
            users.add(user)
            timestamps.add(ts)
            deltats.add(deltat)
        to_read.close()
    similar = open('./data/similar_beijing', 'r').read().strip().split('\n')
    for line in similar:
        user1,rel,user2 = line.split("\t")
        users.add(user1)
        users.add(user2)
    sim.add("similar")
    access.add('access')
    inverse.add('inverse')
    transfer.add('transfer')
    transfer_files = ['train2', 'valid2', 'test2']
    for ff in transfer_files:
        transfer_file = open(os.path.join(path, ff), 'r').read().split('\n')
        for line in transfer_file:
            if len(line)>=5:
                for i in range(1,len(eval(line))):
                    pois.add(eval(line)[i])
    
    users_to_id = {x: i for (i, x) in enumerate(sorted(users))}
    pois_to_id = {x: i for (i, x) in enumerate(sorted(pois))}
    access_to_id = {x: i for (i, x) in enumerate(sorted(access))}
    inverse_to_id = {x: i for (i, x) in enumerate(sorted(inverse))}
    transfer_to_id = {x: i for (i,x) in enumerate(sorted(transfer))}
    timestamps_to_id = {x: i for (i, x) in enumerate(sorted(timestamps))}
    sim_to_id = {x: i for (i, x) in enumerate(sorted(sim))}
    n_relations = len(access) + len(transfer)
    
    print("{} users,{} pois, {} relations over {} timestamps".format(len(users),len(pois),n_relations,len(timestamps)))
    
    if os.path.exists(os.path.join(DATA_PATH, name)):
        shutil.rmtree(os.path.join(DATA_PATH, name))
    os.makedirs(os.path.join(DATA_PATH, name))
    # write ent to id / rel to id
    for (dic, f) in zip([users_to_id, pois_to_id, access_to_id, transfer_to_id, timestamps_to_id,inverse_to_id], ['user_id', 'poi_id', 'access_id', 'transfer_id', 'ts_id','inverse_to_id']):
        ff = open(os.path.join(DATA_PATH, name, f), 'w+')
        for (x, i) in dic.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()

    # map train/test/valid with the ids
    confidence1 = defaultdict(set)
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        examples = []
        for line in to_read.readlines():
            _, origin1, timestamp1, user, loc, timestamp = eval(line)
            ts = tid_96(timestamp)
            ots = tid_96(timestamp1)
            ts1 = time.mktime(time.strptime(timestamp1, "%Y%m%d%H%M"))
            ts2 = time.mktime(time.strptime(timestamp, "%Y%m%d%H%M"))
            deltat =  int((ts2-ts1)//3600)
            examples.append([users_to_id[user],timestamps_to_id[ts], access_to_id['access'], pois_to_id[loc], timestamps_to_id[ots], inverse_to_id['inverse'], pois_to_id[origin1], int(deltat),transfer_to_id['transfer']])
            if f == 'train':
                confidence1[(pois_to_id[origin1],pois_to_id[loc])].add(deltat)

        out = open(Path(DATA_PATH) / name / (f[:-1] + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()
    
    for ff in transfer_files:
        transfer_file = open(os.path.join(path, ff), 'r').read().split('\n')
        transfer_id = []
        for line in transfer_file:
            if len(line)>=5:
                transfer_id.append([users_to_id[eval(line)[0]]] + [pois_to_id[i] for i in eval(line)[1:]])
        out = open(Path(DATA_PATH) / name / (ff + '.pickle'), 'wb')
        pickle.dump(transfer_id, out)
        out.close() 

  
    path1 = []
    for (origin_id,loc_id) in confidence1.keys():
        for deltat in confidence1[(origin_id,loc_id)]:
            path1.append([origin_id,loc_id,deltat])
    
    out = open(Path(DATA_PATH) / name / ('confidence1.pickle'), 'wb')
    pickle.dump(np.array(path1).astype('uint64'), out)
    out.close()


    number = [len(users), len(pois),len(timestamps),len(access),len(inverse),max(deltats), len(transfer)]
    out = open(Path(DATA_PATH) / name / 'numbers.pickle', 'wb')
    pickle.dump(np.array(number).astype('uint64'), out)
    out.close()

    # 访问同地点用户
    similars = []
    for line in similar:
        user1,rel,user2 = line.split("\t")
        similars.append([users_to_id[user1],sim_to_id[rel],users_to_id[user2]])
    out = open(Path(DATA_PATH) / name / ('similar.pickle'), 'wb')
    pickle.dump(np.array(similars).astype('uint64'), out)
    out.close()
    
if __name__ == "__main__":
    datasets = ['kg']
    for d in datasets:
        print("Preparing dataset {}".format(d))
        prepare_dataset(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'data', d
            ),
            d
        )
        

